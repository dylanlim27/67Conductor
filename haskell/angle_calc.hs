import System.IO
import Text.Read (readMaybe)

-- Compute angle in radians at p2 between p1 and p3
computeAngle :: Double -> Double -> Double -> Double -> Double -> Double -> Double
computeAngle x1 y1 x2 y2 x3 y3 =
  let dx1 = x1 - x2
      dy1 = y1 - y2
      dx2 = x3 - x2
      dy2 = y3 - y2
      dot = dx1 * dx2 + dy1 * dy2
      mag1 = sqrt (dx1 * dx1 + dy1 * dy1)
      mag2 = sqrt (dx2 * dx2 + dy2 * dy2)
  in if mag1 == 0.0 || mag2 == 0.0
       then 0.0
       else
         let cosTheta = dot / (mag1 * mag2)
             -- Clamp cosTheta to [-1.0, 1.0] to avoid NaN with acos
             cosClamped = max (-1.0) (min 1.0 cosTheta)
         in acos cosClamped

main :: IO ()
main = do
  hSetBuffering stdin LineBuffering
  hSetBuffering stdout LineBuffering
  loop
  where
    loop = do
      eof <- isEOF
      if eof
        then return ()
        else do
          line <- getLine
          case map readMaybe (words line) of
            [Just x1, Just y1, Just x2, Just y2, Just x3, Just y3] -> do
              let angle = computeAngle x1 y1 x2 y2 x3 y3
              putStrLn (show angle)
              hFlush stdout
              loop
            _ -> do
              putStrLn "0.0"
              hFlush stdout
              loop
