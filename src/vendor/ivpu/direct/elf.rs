//! Bounded, zero-copy Intel device ELF inspection backed by Goblin's ELF definitions.
//!
//! Goblin is used only for device artifacts. Host shared-library loading remains in the existing
//! `dlopen`/`dlsym` boundary because ELF inspection cannot provide loader namespaces, symbol
//! resolution, constructors, TLS, or the host dynamic linker's security semantics.

#![allow(dead_code)]

use goblin::elf::{header, section_header, Elf};

use crate::runtime::artifact::{ArtifactEnvelope, NativePayloadKind, TargetFingerprint};

const ELF64_HEADER_BYTES: usize = goblin::elf64::header::SIZEOF_EHDR;
const ELF64_PROGRAM_HEADER_BYTES: usize = goblin::elf64::program_header::SIZEOF_PHDR;
const ELF64_SECTION_HEADER_BYTES: usize = goblin::elf64::section_header::SIZEOF_SHDR;
const ELF64_SYMBOL_BYTES: u64 = goblin::elf64::sym::SIZEOF_SYM as u64;
const MAX_PROGRAM_HEADERS: usize = 256;
const MAX_SECTION_HEADERS: usize = 4096;
const MAX_SYMBOLS: u64 = 65_536;
const MAX_RELOCATIONS: u64 = 65_536;
pub const MAX_INTEL_RESIDENT_BYTES: u64 = 512 * 1024 * 1024;
pub const INTEL_NPU_MACHINE: u16 = header::EM_NONE;
pub const NPU40XX_ARCH_KIND: u64 = 4;
pub const VPU_SHT_PLATFORM_INFO: u32 = 0x8aaa_aaaf;
pub const VPU_SHT_CMX_METADATA: u32 = 0x8aaa_aaac;
pub const VPU_SHT_CMX_WORKSPACE: u32 = 0x8aaa_aaad;
const VPU_SHF_USER_RESIDENT: u64 = 0x00e0_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElfError {
    Truncated,
    BadMagic,
    UnsupportedClass,
    UnsupportedEndian,
    UnsupportedVersion,
    InvalidHeaderSize,
    TooManyHeaders,
    InvalidTable,
    InvalidRange,
    InvalidAlignment,
    InvalidEntrySize,
    TooManyRelocations,
    TooManySymbols,
    ForeignArchitecture,
    MissingPlatformInfo,
    ResidentLimitExceeded,
    InvalidSymbol,
    InvalidRelocation,
    Goblin,
    WrongPayloadKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElfInspection {
    pub file_type: u16,
    pub machine: u16,
    pub flags: u32,
    pub entry: u64,
    pub program_headers: u16,
    pub section_headers: u16,
    pub relocation_count: u64,
    pub symbol_count: u64,
    pub loadable_file_bytes: u64,
    pub loadable_memory_bytes: u64,
    pub resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceSection<'a> {
    pub index: usize,
    pub name: Option<&'a str>,
    pub section_type: u32,
    pub flags: u64,
    pub address: u64,
    pub alignment: u64,
    pub memory_bytes: u64,
    pub data: Option<&'a [u8]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceSymbol<'a> {
    pub table_section: usize,
    pub index: usize,
    pub name: Option<&'a str>,
    pub section_index: usize,
    pub value: u64,
    pub bytes: u64,
    pub binding: u8,
    pub symbol_type: u8,
    pub visibility: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceRelocation {
    pub relocation_section: usize,
    pub target_section: usize,
    pub symbol_table: usize,
    pub index: usize,
    pub offset: u64,
    pub symbol: usize,
    pub relocation_type: u32,
    pub addend: Option<i64>,
    pub jit: bool,
}

pub struct ParsedIntelElf<'a> {
    bytes: &'a [u8],
    elf: Elf<'a>,
    inspection: ElfInspection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindingMetadata {
    pub abi_major: u16,
    pub abi_minor: u16,
    pub inputs: u16,
    pub outputs: u16,
    pub scratch: u16,
    pub resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactInspection {
    pub elf: ElfInspection,
    pub bindings: BindingMetadata,
    pub target: TargetFingerprint,
}

pub struct ParsedIntelArtifact<'a> {
    pub elf: ParsedIntelElf<'a>,
    pub bindings: BindingMetadata,
    pub target: TargetFingerprint,
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ElfError> {
    let value = bytes.get(offset..offset + 2).ok_or(ElfError::Truncated)?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ElfError> {
    let value = bytes.get(offset..offset + 4).ok_or(ElfError::Truncated)?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, ElfError> {
    let value = bytes.get(offset..offset + 8).ok_or(ElfError::Truncated)?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

fn table_range(
    offset: u64,
    count: usize,
    entry_bytes: usize,
    file_bytes: usize,
) -> Result<core::ops::Range<usize>, ElfError> {
    let offset = usize::try_from(offset).map_err(|_| ElfError::InvalidTable)?;
    let bytes = count
        .checked_mul(entry_bytes)
        .ok_or(ElfError::InvalidTable)?;
    let end = offset.checked_add(bytes).ok_or(ElfError::InvalidTable)?;
    if end > file_bytes {
        return Err(ElfError::InvalidTable);
    }
    Ok(offset..end)
}

fn validate_file_range(offset: u64, bytes: u64, file_bytes: usize) -> Result<(), ElfError> {
    let end = offset.checked_add(bytes).ok_or(ElfError::InvalidRange)?;
    if end > file_bytes as u64 {
        return Err(ElfError::InvalidRange);
    }
    Ok(())
}

fn section_has_file_data(section_type: u32) -> bool {
    !matches!(
        section_type,
        section_header::SHT_NOBITS | VPU_SHT_CMX_METADATA | VPU_SHT_CMX_WORKSPACE
    )
}

pub fn inspect_elf(bytes: &[u8]) -> Result<ElfInspection, ElfError> {
    if bytes.len() < ELF64_HEADER_BYTES {
        return Err(ElfError::Truncated);
    }
    if bytes[..4] != [0x7f, b'E', b'L', b'F'] {
        return Err(ElfError::BadMagic);
    }
    if bytes[4] != header::ELFCLASS64 {
        return Err(ElfError::UnsupportedClass);
    }
    if bytes[5] != header::ELFDATA2LSB {
        return Err(ElfError::UnsupportedEndian);
    }
    if bytes[6] != 0 && bytes[6] != header::EV_CURRENT as u8 {
        return Err(ElfError::UnsupportedVersion);
    }
    let parsed_header = Elf::parse_header(bytes).map_err(|_| ElfError::Goblin)?;
    if (parsed_header.e_version != 0 && parsed_header.e_version != u32::from(header::EV_CURRENT))
        || parsed_header.e_ehsize as usize != ELF64_HEADER_BYTES
    {
        return Err(ElfError::InvalidHeaderSize);
    }

    let phoff = parsed_header.e_phoff;
    let shoff = parsed_header.e_shoff;
    let phentsize = parsed_header.e_phentsize as usize;
    let phnum = parsed_header.e_phnum as usize;
    let shentsize = parsed_header.e_shentsize as usize;
    let shnum = parsed_header.e_shnum as usize;
    let shstrndx = parsed_header.e_shstrndx as usize;

    if phnum > MAX_PROGRAM_HEADERS || shnum > MAX_SECTION_HEADERS {
        return Err(ElfError::TooManyHeaders);
    }
    if phnum != 0 && phentsize != ELF64_PROGRAM_HEADER_BYTES {
        return Err(ElfError::InvalidEntrySize);
    }
    if shnum != 0 && shentsize != ELF64_SECTION_HEADER_BYTES {
        return Err(ElfError::InvalidEntrySize);
    }
    if shnum != 0 && shstrndx >= shnum {
        return Err(ElfError::InvalidTable);
    }

    let phdrs = table_range(phoff, phnum, phentsize, bytes.len())?;
    let shdrs = table_range(shoff, shnum, shentsize, bytes.len())?;
    let mut loadable_file_bytes = 0u64;
    let mut loadable_memory_bytes = 0u64;
    for index in 0..phnum {
        let base = phdrs.start + index * phentsize;
        let offset = read_u64(bytes, base + 8)?;
        let file_size = read_u64(bytes, base + 32)?;
        let memory_size = read_u64(bytes, base + 40)?;
        let alignment = read_u64(bytes, base + 48)?;
        if file_size > memory_size {
            return Err(ElfError::InvalidRange);
        }
        if alignment > 1 && !alignment.is_power_of_two() {
            return Err(ElfError::InvalidAlignment);
        }
        validate_file_range(offset, file_size, bytes.len())?;
        loadable_file_bytes = loadable_file_bytes
            .checked_add(file_size)
            .ok_or(ElfError::InvalidRange)?;
        loadable_memory_bytes = loadable_memory_bytes
            .checked_add(memory_size)
            .ok_or(ElfError::InvalidRange)?;
    }

    let mut relocation_count = 0u64;
    let mut symbol_count = 0u64;
    let mut resident_bytes = 0u64;
    for index in 0..shnum {
        let base = shdrs.start + index * shentsize;
        let section_type = read_u32(bytes, base + 4)?;
        let offset = read_u64(bytes, base + 24)?;
        let size = read_u64(bytes, base + 32)?;
        let alignment = read_u64(bytes, base + 48)?;
        let entry_size = read_u64(bytes, base + 56)?;
        if alignment > 1 && !alignment.is_power_of_two() {
            return Err(ElfError::InvalidAlignment);
        }
        if section_has_file_data(section_type) {
            validate_file_range(offset, size, bytes.len())?;
        }
        if read_u64(bytes, base + 8)?
            & (u64::from(section_header::SHF_ALLOC) | VPU_SHF_USER_RESIDENT)
            != 0
            || matches!(section_type, VPU_SHT_CMX_METADATA | VPU_SHT_CMX_WORKSPACE)
        {
            resident_bytes = resident_bytes
                .checked_add(size)
                .ok_or(ElfError::ResidentLimitExceeded)?;
            if resident_bytes > MAX_INTEL_RESIDENT_BYTES {
                return Err(ElfError::ResidentLimitExceeded);
            }
        }
        if section_type == section_header::SHT_RELA || section_type == section_header::SHT_REL {
            let expected = if section_type == section_header::SHT_RELA {
                24
            } else {
                16
            };
            if entry_size != expected || size % entry_size != 0 {
                return Err(ElfError::InvalidEntrySize);
            }
            relocation_count = relocation_count
                .checked_add(size / entry_size)
                .ok_or(ElfError::TooManyRelocations)?;
            if relocation_count > MAX_RELOCATIONS {
                return Err(ElfError::TooManyRelocations);
            }
        }
        if section_type == section_header::SHT_SYMTAB || section_type == section_header::SHT_DYNSYM
        {
            if entry_size != ELF64_SYMBOL_BYTES || size % entry_size != 0 {
                return Err(ElfError::InvalidEntrySize);
            }
            symbol_count = symbol_count
                .checked_add(size / entry_size)
                .ok_or(ElfError::TooManySymbols)?;
            if symbol_count > MAX_SYMBOLS {
                return Err(ElfError::TooManySymbols);
            }
            let string_table = read_u32(bytes, base + 40)? as usize;
            if string_table >= shnum {
                return Err(ElfError::InvalidSymbol);
            }
        }
    }

    Ok(ElfInspection {
        file_type: parsed_header.e_type,
        machine: parsed_header.e_machine,
        flags: parsed_header.e_flags,
        entry: parsed_header.e_entry,
        program_headers: phnum as u16,
        section_headers: shnum as u16,
        relocation_count,
        symbol_count,
        loadable_file_bytes,
        loadable_memory_bytes,
        resident_bytes,
    })
}

fn section_header_offset(parsed_header: &header::Header, index: usize) -> Result<usize, ElfError> {
    let base = usize::try_from(parsed_header.e_shoff).map_err(|_| ElfError::InvalidTable)?;
    base.checked_add(
        index
            .checked_mul(parsed_header.e_shentsize as usize)
            .ok_or(ElfError::InvalidTable)?,
    )
    .ok_or(ElfError::InvalidTable)
}

fn section_data<'a>(
    bytes: &'a [u8],
    parsed_header: &header::Header,
    index: usize,
) -> Result<Option<&'a [u8]>, ElfError> {
    if index >= parsed_header.e_shnum as usize {
        return Err(ElfError::InvalidTable);
    }
    let base = section_header_offset(parsed_header, index)?;
    if !section_has_file_data(read_u32(bytes, base + 4)?) {
        return Ok(None);
    }
    let offset =
        usize::try_from(read_u64(bytes, base + 24)?).map_err(|_| ElfError::InvalidRange)?;
    let size = usize::try_from(read_u64(bytes, base + 32)?).map_err(|_| ElfError::InvalidRange)?;
    let end = offset.checked_add(size).ok_or(ElfError::InvalidRange)?;
    bytes
        .get(offset..end)
        .map(Some)
        .ok_or(ElfError::InvalidRange)
}

fn string_at(bytes: &[u8], offset: usize) -> Option<&str> {
    let tail = bytes.get(offset..)?;
    let end = tail.iter().position(|&byte| byte == 0)?;
    core::str::from_utf8(&tail[..end]).ok()
}

fn platform_architecture(bytes: &[u8], parsed_header: &header::Header) -> Result<u64, ElfError> {
    let mut architecture = None;
    for index in 0..parsed_header.e_shnum as usize {
        let base = section_header_offset(parsed_header, index)?;
        if read_u32(bytes, base + 4)? != VPU_SHT_PLATFORM_INFO {
            continue;
        }
        if architecture.is_some() {
            return Err(ElfError::ForeignArchitecture);
        }
        let data =
            section_data(bytes, parsed_header, index)?.ok_or(ElfError::MissingPlatformInfo)?;
        // PlatformInfoSerialization in npu_compiler_elf starts with one 32-byte
        // SerialDescriptor, followed by one 8-byte ArchKind element.
        if data.len() < 40
            || read_u64(data, 8)? != 0
            || read_u64(data, 16)? != 1
            || read_u64(data, 24)? != 8
        {
            return Err(ElfError::MissingPlatformInfo);
        }
        let data_offset =
            usize::try_from(read_u64(data, 0)?).map_err(|_| ElfError::MissingPlatformInfo)?;
        architecture = Some(read_u64(data, data_offset)?);
    }
    architecture.ok_or(ElfError::MissingPlatformInfo)
}

fn validate_symbol_relocation_bounds(
    bytes: &[u8],
    parsed_header: &header::Header,
) -> Result<(), ElfError> {
    let section_count = parsed_header.e_shnum as usize;
    for index in 0..section_count {
        let base = section_header_offset(parsed_header, index)?;
        let section_type = read_u32(bytes, base + 4)?;
        let section_offset =
            usize::try_from(read_u64(bytes, base + 24)?).map_err(|_| ElfError::InvalidRange)?;
        let section_size = read_u64(bytes, base + 32)?;
        let linked = read_u32(bytes, base + 40)? as usize;
        let target = read_u32(bytes, base + 44)? as usize;
        let entry_size = read_u64(bytes, base + 56)?;
        if matches!(
            section_type,
            section_header::SHT_SYMTAB | section_header::SHT_DYNSYM
        ) {
            let strings =
                section_data(bytes, parsed_header, linked)?.ok_or(ElfError::InvalidSymbol)?;
            for symbol in 0..(section_size / entry_size) as usize {
                let symbol_base = section_offset + symbol * entry_size as usize;
                let name = read_u32(bytes, symbol_base)? as usize;
                let section_index = read_u16(bytes, symbol_base + 6)? as usize;
                if name >= strings.len()
                    || (section_index < 0xff00 && section_index >= section_count)
                {
                    return Err(ElfError::InvalidSymbol);
                }
            }
        }
        if matches!(
            section_type,
            section_header::SHT_REL | section_header::SHT_RELA
        ) {
            if linked >= section_count || target >= section_count {
                return Err(ElfError::InvalidRelocation);
            }
            let symbol_base = section_header_offset(parsed_header, linked)?;
            let symbol_type = read_u32(bytes, symbol_base + 4)?;
            let symbol_bytes = read_u64(bytes, symbol_base + 32)?;
            let symbol_entry = read_u64(bytes, symbol_base + 56)?;
            if !matches!(
                symbol_type,
                section_header::SHT_SYMTAB | section_header::SHT_DYNSYM
            ) || symbol_entry != ELF64_SYMBOL_BYTES
            {
                return Err(ElfError::InvalidRelocation);
            }
            let symbols = symbol_bytes / symbol_entry;
            let target_base = section_header_offset(parsed_header, target)?;
            let target_bytes = read_u64(bytes, target_base + 32)?;
            for relocation in 0..(section_size / entry_size) as usize {
                let relocation_base = section_offset + relocation * entry_size as usize;
                let offset = read_u64(bytes, relocation_base)?;
                let info = read_u64(bytes, relocation_base + 8)?;
                if info >> 32 >= symbols
                    || offset.checked_add(8).is_none()
                    || offset + 8 > target_bytes
                {
                    return Err(ElfError::InvalidRelocation);
                }
            }
        }
    }
    Ok(())
}

pub fn parse_intel_elf(bytes: &[u8]) -> Result<ParsedIntelElf<'_>, ElfError> {
    let inspection = inspect_elf(bytes)?;
    if inspection.machine != INTEL_NPU_MACHINE
        || !matches!(inspection.file_type, header::ET_REL | header::ET_EXEC)
    {
        return Err(ElfError::ForeignArchitecture);
    }
    let parsed_header = Elf::parse_header(bytes).map_err(|_| ElfError::Goblin)?;
    if platform_architecture(bytes, &parsed_header)? != NPU40XX_ARCH_KIND {
        return Err(ElfError::ForeignArchitecture);
    }
    validate_symbol_relocation_bounds(bytes, &parsed_header)?;

    // Goblin's unified parser allocates header vectors. Every count, table range, symbol count,
    // relocation count, and resident byte total has already passed the strict preflight limits.
    // Payload bytes, section data, names, symbols, and relocations remain borrowed/zero-copy.
    let elf = Elf::parse(bytes).map_err(|_| ElfError::Goblin)?;
    if !elf.is_64 || !elf.little_endian {
        return Err(ElfError::ForeignArchitecture);
    }
    let parsed = ParsedIntelElf {
        bytes,
        elf,
        inspection,
    };
    parsed.validate_symbols_and_relocations()?;
    Ok(parsed)
}

impl<'a> ParsedIntelElf<'a> {
    pub const fn inspection(&self) -> &ElfInspection {
        &self.inspection
    }

    pub fn visit_sections(&self, mut visitor: impl FnMut(DeviceSection<'a>)) {
        for (index, section) in self.elf.section_headers.iter().enumerate() {
            let data = if !section_has_file_data(section.sh_type) {
                None
            } else {
                let start = section.sh_offset as usize;
                let end = start + section.sh_size as usize;
                self.bytes.get(start..end)
            };
            visitor(DeviceSection {
                index,
                name: self.elf.shdr_strtab.get_at(section.sh_name),
                section_type: section.sh_type,
                flags: section.sh_flags,
                address: section.sh_addr,
                alignment: section.sh_addralign,
                memory_bytes: section.sh_size,
                data,
            });
        }
    }

    pub fn find_section(&self, section_index: usize) -> Result<DeviceSection<'a>, ElfError> {
        let mut found = None;
        self.visit_sections(|section| {
            if section.index == section_index {
                found = Some(section);
            }
        });
        found.ok_or(ElfError::InvalidTable)
    }

    pub fn visit_symbols(&self, mut visitor: impl FnMut(DeviceSymbol<'a>)) -> Result<(), ElfError> {
        for (table_section, section) in self.elf.section_headers.iter().enumerate() {
            if !matches!(
                section.sh_type,
                section_header::SHT_SYMTAB | section_header::SHT_DYNSYM
            ) {
                continue;
            }
            let strings = section_data(self.bytes, &self.elf.header, section.sh_link as usize)?
                .ok_or(ElfError::InvalidSymbol)?;
            let count = section.sh_size / ELF64_SYMBOL_BYTES;
            for index in 0..count as usize {
                let base = section.sh_offset as usize + index * ELF64_SYMBOL_BYTES as usize;
                let name_offset = read_u32(self.bytes, base)? as usize;
                let info = *self.bytes.get(base + 4).ok_or(ElfError::InvalidSymbol)?;
                let visibility = *self.bytes.get(base + 5).ok_or(ElfError::InvalidSymbol)?;
                visitor(DeviceSymbol {
                    table_section,
                    index,
                    name: string_at(strings, name_offset),
                    section_index: read_u16(self.bytes, base + 6)? as usize,
                    value: read_u64(self.bytes, base + 8)?,
                    bytes: read_u64(self.bytes, base + 16)?,
                    binding: info >> 4,
                    symbol_type: info & 0x0f,
                    visibility,
                });
            }
        }
        Ok(())
    }

    pub fn find_symbol(
        &self,
        table_section: usize,
        symbol_index: usize,
    ) -> Result<DeviceSymbol<'a>, ElfError> {
        let mut found = None;
        self.visit_symbols(|symbol| {
            if symbol.table_section == table_section && symbol.index == symbol_index {
                found = Some(symbol);
            }
        })?;
        found.ok_or(ElfError::InvalidSymbol)
    }

    pub fn visit_relocations(
        &self,
        mut visitor: impl FnMut(DeviceRelocation),
    ) -> Result<(), ElfError> {
        for (relocation_section, section) in self.elf.section_headers.iter().enumerate() {
            if !matches!(
                section.sh_type,
                section_header::SHT_REL | section_header::SHT_RELA
            ) {
                continue;
            }
            let is_rela = section.sh_type == section_header::SHT_RELA;
            let count = section.sh_size / section.sh_entsize;
            for index in 0..count as usize {
                let base = section.sh_offset as usize + index * section.sh_entsize as usize;
                let info = read_u64(self.bytes, base + 8)?;
                visitor(DeviceRelocation {
                    relocation_section,
                    target_section: section.sh_info as usize,
                    symbol_table: section.sh_link as usize,
                    index,
                    offset: read_u64(self.bytes, base)?,
                    symbol: (info >> 32) as usize,
                    relocation_type: info as u32,
                    addend: if is_rela {
                        Some(read_u64(self.bytes, base + 16)? as i64)
                    } else {
                        None
                    },
                    jit: section.sh_flags & 0x0010_0000 != 0,
                });
            }
        }
        Ok(())
    }

    fn validate_symbols_and_relocations(&self) -> Result<(), ElfError> {
        self.visit_symbols(|symbol| {
            let _ = symbol;
        })?;
        let mut invalid = false;
        self.visit_relocations(|relocation| {
            let Some(target) = self.elf.section_headers.get(relocation.target_section) else {
                invalid = true;
                return;
            };
            let Some(table) = self
                .elf
                .section_headers
                .get(self.elf.section_headers[relocation.relocation_section].sh_link as usize)
            else {
                invalid = true;
                return;
            };
            if !matches!(
                table.sh_type,
                section_header::SHT_SYMTAB | section_header::SHT_DYNSYM
            ) || relocation.symbol as u64 >= table.sh_size / ELF64_SYMBOL_BYTES
                || relocation.offset.checked_add(8).is_none()
                || relocation.offset + 8 > target.sh_size
            {
                invalid = true;
            }
        })?;
        if invalid {
            return Err(ElfError::InvalidRelocation);
        }
        Ok(())
    }
}

pub fn inspect_artifact(artifact: &ArtifactEnvelope<'_>) -> Result<ArtifactInspection, ElfError> {
    let parsed = parse_intel_artifact(artifact)?;
    Ok(ArtifactInspection {
        elf: parsed.elf.inspection,
        bindings: parsed.bindings,
        target: parsed.target,
    })
}

pub fn parse_intel_artifact<'a>(
    artifact: &ArtifactEnvelope<'a>,
) -> Result<ParsedIntelArtifact<'a>, ElfError> {
    if artifact.header().payload_kind != NativePayloadKind::INTEL_IVPU_ELF {
        return Err(ElfError::WrongPayloadKind);
    }
    if artifact.header().resident_bytes > MAX_INTEL_RESIDENT_BYTES {
        return Err(ElfError::ResidentLimitExceeded);
    }
    let abi = artifact.header().binding_abi;
    let elf = parse_intel_elf(artifact.payload())?;
    if elf.inspection.resident_bytes > artifact.header().resident_bytes {
        return Err(ElfError::ResidentLimitExceeded);
    }
    Ok(ParsedIntelArtifact {
        elf,
        bindings: BindingMetadata {
            abi_major: abi.major,
            abi_minor: abi.minor,
            inputs: abi.input_count,
            outputs: abi.output_count,
            scratch: abi.scratch_count,
            resident_bytes: artifact.header().resident_bytes,
        },
        target: artifact.header().target,
    })
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "bench-hardware")]
    use std::io::Read;

    use super::*;

    fn minimal_elf() -> [u8; 64] {
        let mut elf = [0u8; 64];
        elf[..7].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1]);
        elf[16..18].copy_from_slice(&2u16.to_le_bytes());
        elf[20..24].copy_from_slice(&1u32.to_le_bytes());
        elf[52..54].copy_from_slice(&64u16.to_le_bytes());
        elf
    }

    fn minimal_intel_elf() -> alloc::vec::Vec<u8> {
        let mut elf = alloc::vec![0u8; 320];
        elf[..7].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1]);
        elf[16..18].copy_from_slice(&header::ET_REL.to_le_bytes());
        elf[18..20].copy_from_slice(&INTEL_NPU_MACHINE.to_le_bytes());
        elf[20..24].copy_from_slice(&1u32.to_le_bytes());
        elf[40..48].copy_from_slice(&64u64.to_le_bytes());
        elf[52..54].copy_from_slice(&64u16.to_le_bytes());
        elf[58..60].copy_from_slice(&64u16.to_le_bytes());
        elf[60..62].copy_from_slice(&3u16.to_le_bytes());
        elf[62..64].copy_from_slice(&2u16.to_le_bytes());

        let platform = 64 + 64;
        elf[platform + 4..platform + 8].copy_from_slice(&VPU_SHT_PLATFORM_INFO.to_le_bytes());
        elf[platform + 24..platform + 32].copy_from_slice(&256u64.to_le_bytes());
        elf[platform + 32..platform + 40].copy_from_slice(&40u64.to_le_bytes());
        elf[platform + 48..platform + 56].copy_from_slice(&8u64.to_le_bytes());
        elf[256..264].copy_from_slice(&32u64.to_le_bytes());
        elf[272..280].copy_from_slice(&1u64.to_le_bytes());
        elf[280..288].copy_from_slice(&8u64.to_le_bytes());
        elf[288..296].copy_from_slice(&NPU40XX_ARCH_KIND.to_le_bytes());

        let strings = 64 + 128;
        elf[strings + 4..strings + 8].copy_from_slice(&section_header::SHT_STRTAB.to_le_bytes());
        elf[strings + 24..strings + 32].copy_from_slice(&296u64.to_le_bytes());
        elf[strings + 32..strings + 40].copy_from_slice(&1u64.to_le_bytes());
        elf[strings + 48..strings + 56].copy_from_slice(&1u64.to_le_bytes());
        elf
    }

    #[test]
    fn bounded_elf_inspector_accepts_minimal_header() {
        let inspection = inspect_elf(&minimal_elf()).unwrap();
        assert_eq!(inspection.program_headers, 0);
        assert_eq!(inspection.relocation_count, 0);
    }

    #[test]
    fn malformed_elf_is_rejected_without_allocation() {
        assert_eq!(inspect_elf(&[]), Err(ElfError::Truncated));
        let mut elf = minimal_elf();
        elf[40..48].copy_from_slice(&63u64.to_le_bytes());
        elf[58..60].copy_from_slice(&64u16.to_le_bytes());
        elf[60..62].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(inspect_elf(&elf), Err(ElfError::InvalidTable));
    }

    #[test]
    fn goblin_parser_accepts_pinned_npu40xx_target() {
        let elf = minimal_intel_elf();
        let parsed = parse_intel_elf(&elf).unwrap();
        assert_eq!(parsed.inspection().machine, INTEL_NPU_MACHINE);
        let mut sections = 0;
        parsed.visit_sections(|_| sections += 1);
        assert_eq!(sections, 3);
    }

    #[test]
    fn foreign_class_machine_and_resident_overflow_are_rejected() {
        let mut elf = minimal_intel_elf();
        elf[4] = header::ELFCLASS32;
        assert!(matches!(
            parse_intel_elf(&elf),
            Err(ElfError::UnsupportedClass)
        ));

        let mut elf = minimal_intel_elf();
        elf[18..20].copy_from_slice(&header::EM_X86_64.to_le_bytes());
        assert!(matches!(
            parse_intel_elf(&elf),
            Err(ElfError::ForeignArchitecture)
        ));

        let mut elf = minimal_intel_elf();
        elf[60..62].copy_from_slice(&4u16.to_le_bytes());
        let oversized = 64 + 192;
        elf[oversized + 4..oversized + 8]
            .copy_from_slice(&section_header::SHT_NOBITS.to_le_bytes());
        elf[oversized + 8..oversized + 16]
            .copy_from_slice(&u64::from(section_header::SHF_ALLOC).to_le_bytes());
        elf[oversized + 32..oversized + 40]
            .copy_from_slice(&(MAX_INTEL_RESIDENT_BYTES + 1).to_le_bytes());
        elf[oversized + 48..oversized + 56].copy_from_slice(&8u64.to_le_bytes());
        assert!(matches!(
            parse_intel_elf(&elf),
            Err(ElfError::ResidentLimitExceeded)
        ));
    }

    #[test]
    #[cfg(feature = "bench-hardware")]
    #[ignore = "local cached Intel artifact inspection only; never loads or submits"]
    fn parse_existing_tiny_intel_artifact() {
        const PATH: &str = "/home/cassmantle/.cache/ggml-npu/ir2-8c91da4914adc7d5-n1.blob";
        const MAX_TEST_ARTIFACT: usize = 1024 * 1024;
        let metadata = std::fs::metadata(PATH).unwrap();
        let length = usize::try_from(metadata.len()).unwrap();
        assert!(length <= MAX_TEST_ARTIFACT);
        let mut bytes = alloc::vec::Vec::new();
        bytes.try_reserve_exact(length).unwrap();
        bytes.resize(length, 0);
        std::fs::File::open(PATH)
            .unwrap()
            .read_exact(&mut bytes)
            .unwrap();
        let parsed = parse_intel_elf(&bytes).unwrap();
        let mut firmware_api = [0u64; 8];
        firmware_api[super::super::firmware::JOB_CMD_API_INDEX] =
            super::super::firmware::JOB_CMD_API_WORD as u64;
        firmware_api[super::super::firmware::NNRT_API_INDEX] =
            super::super::firmware::NNRT_API_WORD as u64;
        let device = super::super::DeviceParameters {
            device_id: 0x643e,
            revision: 4,
            platform_type: 0,
            core_clock_rate: 1_950_000_000,
            context_id: 1,
            context_base_address: 0x1_0000_0000,
            firmware_api,
            tile_config: 0x20,
            sku: 0,
            capability_bits: 0x1e,
            max_contexts: 64,
            preempt_buffer_bytes: 28_389_376,
        };
        let metadata = super::super::hpi::inspect_metadata(&parsed).unwrap();
        let mut input_sections = 0usize;
        let mut output_sections = 0usize;
        let mut profiling_sections = 0usize;
        super::super::link::visit_binding_requirements(&parsed, |binding| match binding.class {
            super::super::link::DeviceMemoryClass::Input => input_sections += 1,
            super::super::link::DeviceMemoryClass::Output => output_sections += 1,
            super::super::link::DeviceMemoryClass::Profiling => profiling_sections += 1,
            _ => {}
        });
        std::println!(
            "metadata_inputs={} metadata_outputs={} metadata_profiling={} sections_inputs={input_sections} sections_outputs={output_sections} sections_profiling={profiling_sections}",
            metadata.inputs,
            metadata.outputs,
            metadata.profiling_outputs,
        );
        let hpi = super::super::hpi::inspect_hpi_requirements(&parsed, &device).unwrap();
        let mut link_operations = 0usize;
        let mut jit_operations = 0usize;
        let link_status = super::super::link::visit_link_operations(&parsed, |operation| {
            link_operations += 1;
            jit_operations += usize::from(operation.jit);
        });
        let mut placements = 0usize;
        let mut placement_bytes = 0u64;
        super::super::link::visit_section_placements(&parsed, |placement| {
            placements += 1;
            placement_bytes += placement.bytes;
        });
        std::println!(
            "artifact={PATH} bytes={length} sections={} symbols={} relocations={} resident_bytes={} placements={placements} placement_bytes={placement_bytes} inputs={} outputs={} cmx_slice_bytes={} slices={} barriers={} loader={}.{}.{} nnrt={}.{}.{} entry_section={} entry_bytes={} link_operations={link_operations} jit_operations={jit_operations} controlled_link={link_status:?}",
            parsed.inspection().section_headers,
            parsed.inspection().symbol_count,
            parsed.inspection().relocation_count,
            parsed.inspection().resident_bytes,
            hpi.resources.inputs,
            hpi.resources.outputs,
            hpi.resources.cmx_slice_bytes,
            hpi.resources.slice_count,
            hpi.resources.barriers,
            hpi.versions.loader_major,
            hpi.versions.loader_minor,
            hpi.versions.loader_patch,
            hpi.versions.mapped_inference_major,
            hpi.versions.mapped_inference_minor,
            hpi.versions.mapped_inference_patch,
            hpi.entry_section,
            hpi.entry_bytes,
        );
        assert!(link_status.is_ok());
        assert_eq!(link_operations as u64, parsed.inspection().relocation_count);
    }
}
