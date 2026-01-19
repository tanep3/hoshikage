use crate::error::Result;
use libloading::{Library, Symbol};
use std::path::PathBuf;

pub struct DynamicLibraryLoader {
    lib: Option<Library>,
    lib_path: PathBuf,
}

impl DynamicLibraryLoader {
    pub fn new(lib_path: PathBuf) -> Self {
        Self {
            lib: None,
            lib_path,
        }
    }

    pub fn load(&mut self) -> Result<()> {
        let lib = unsafe {
            Library::new(&self.lib_path).map_err(|e| {
                crate::error::HoshikageError::LibraryLoadError(format!(
                    "Failed to load library from {}: {}",
                    self.lib_path.display(),
                    e
                ))
            })?
        };

        self.lib = Some(lib);
        Ok(())
    }

    pub fn is_loaded(&self) -> bool {
        self.lib.is_some()
    }

    /// # Safety
    ///
    /// This function is unsafe because it loads a symbol from a dynamic library.
    /// The caller must ensure that:
    /// - The library is loaded before calling this function
    /// - The symbol name is valid and exists in the library
    /// - The type T matches the actual symbol type in the library
    pub unsafe fn get_symbol<T>(&self, name: &str) -> Result<Symbol<'_, T>> {
        let lib = self.lib.as_ref().ok_or_else(|| {
            crate::error::HoshikageError::LibraryLoadError("Library not loaded".to_string())
        })?;

        lib.get(name.as_bytes()).map_err(|e| {
            crate::error::HoshikageError::LibraryLoadError(format!(
                "Failed to get symbol '{}': {}",
                name, e
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_loader() {
        let loader = DynamicLibraryLoader::new(PathBuf::from("test.so"));
        assert!(!loader.is_loaded());
    }

    #[test]
    fn test_not_loaded() {
        let loader = DynamicLibraryLoader::new(PathBuf::from("test.so"));
        assert!(!loader.is_loaded());
    }

    #[test]
    fn test_lib_path() {
        let path = PathBuf::from("/path/to/libllama.so");
        let loader = DynamicLibraryLoader::new(path.clone());
        assert_eq!(loader.lib_path, path);
    }
}
