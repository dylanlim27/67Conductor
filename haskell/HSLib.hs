{-# LANGUAGE ForeignFunctionInterface #-}
module HSLib where

import Foreign.C.Types
import Foreign.C.String
import Foreign.Marshal.Alloc (free)

foreign export ccall hs_add :: CInt -> CInt -> CInt
foreign export ccall hs_reverse :: CString -> IO CString
foreign export ccall hs_free :: CString -> IO ()

hs_add :: CInt -> CInt -> CInt
hs_add a b = a + b

hs_reverse :: CString -> IO CString
hs_reverse cstr = do
  str <- peekCString cstr
  newCString (reverse str)

hs_free :: CString -> IO ()
hs_free cstr = free cstr
