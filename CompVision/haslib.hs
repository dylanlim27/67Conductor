{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
module Main where

import qualified Data.ByteString.Char8 as B
import qualified Data.ByteString.Lazy.Char8 as BL
import Network.Socket
import Control.Monad (forever)
import Data.Aeson
import GHC.Generics
import qualified Data.Map.Strict as M
import Text.Regex.TDFA ((=~))
import System.IO

data LandmarkPoint = LandmarkPoint
  { x :: Double
  , y :: Double
  , z :: Double
  } deriving (Show, Generic)

instance FromJSON LandmarkPoint

data ObjectWrapper = ObjectWrapper
  { ts :: Double
  , landmarks :: M.Map String LandmarkPoint
  } deriving (Show, Generic)

instance FromJSON ObjectWrapper

classifyAB :: M.Map String LandmarkPoint -> Maybe Char
classifyAB lms = do
  l_sh <- y <$> M.lookup "11" lms
  r_sh <- y <$> M.lookup "12" lms
  l_wr <- y <$> M.lookup "15" lms
  r_wr <- y <$> M.lookup "16" lms
  let up_margin = 0.03
      down_margin = 0.03
      left_up = l_wr < (l_sh - up_margin)
      left_down = l_wr > (l_sh + down_margin)
      right_up = r_wr < (r_sh - up_margin)
      right_down = r_wr > (r_sh + down_margin)
  if left_up && right_down
    then Just 'A'
    else if left_down && right_up
           then Just 'B'
           else Nothing

sendNotify :: String -> Int -> String -> IO ()
sendNotify host port msg = do
  addrinfos <- getAddrInfo (Just (defaultHints { addrSocketType = Datagram })) (Just host) (Just (show port))
  let serveraddr = head addrinfos
  sock <- socket (addrFamily serveraddr) Datagram defaultProtocol
  _ <- sendTo sock (B.pack msg) (addrAddress serveraddr)
  close sock

main :: IO ()
main = withSocketsDo $ do
  let pythonHost = "127.0.0.1"
      pythonNotifyPort = 5006
      listenPort = 5005
  addrinfos <- getAddrInfo (Just (defaultHints { addrFlags = [AI_PASSIVE], addrSocketType = Datagram })) Nothing (Just (show listenPort))
  let serveraddr = head addrinfos
  sock <- socket (addrFamily serveraddr) Datagram defaultProtocol
  bind sock (addrAddress serveraddr)
  putStrLn $ "Listening for frames on UDP port " ++ show listenPort
  hFlush stdout
  loop sock ""
  where
    pythonHost = "127.0.0.1"
    pythonNotifyPort = 5006
    loop sock seqStr = do
      (msg, _) <- recvFrom sock 4096
      let mobj = decode (BL.fromStrict msg) :: Maybe ObjectWrapper
      case mobj of
        Nothing -> do
          putStrLn "bad json"
          hFlush stdout
          loop sock seqStr
        Just obj -> do
          case classifyAB (landmarks obj) of
            Nothing -> do
              loop sock seqStr
            Just code -> do
              let newSeq = if null seqStr || last seqStr /= code
                             then seqStr ++ [code]
                             else seqStr
              putStrLn $ "Got code: " ++ [code] ++ " seq=" ++ newSeq
              hFlush stdout
              let regex = "^(?:AB){2,}(?:A)?$|^(?:BA){2,}(?:B)?$" :: String
                  matched = newSeq =~ regex :: Bool
              if matched
                then do
                  putStrLn $ "Pattern matched: " ++ newSeq
                  hFlush stdout
                  sendNotify pythonHost pythonNotifyPort ("PATTERN:" ++ newSeq)
                  loop sock ""
                else do
                  let trimmedSeq = if length newSeq > 10 then drop (length newSeq - 10) newSeq else newSeq
                  loop sock trimmedSeq