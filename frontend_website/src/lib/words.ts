export default function name123(params: string[] = ["dress", "hat", "longsleeve", "outwear", "pants", "shirt", "shoes", "shorts", "skirt", "t-shirt", "casual", "formal", "sportswear", "vintage", "minimalist", "streetwear"]) {
  return JSON.stringify({
      words: params,
    })
}