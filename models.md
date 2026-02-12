# Model Inventory

Location: `/models/`

## Checkpoints

### SD 1.5 (512x512)

| Model | Size | GGUF | Tensors | Notes |
|-------|------|------|---------|-------|
| dreamshaper_8 | 2.0G | 1.7G | 1,131 | |
| epicrealism_naturalSinRC1VAE | 2.0G | 1.7G | 1,133 | Includes VAE |
| hassakuSD15_v13 | 2.0G | 1.7G | 1,131 | |

### SDXL / Pony / Illustrious (1024x1024)

| Model | Size | GGUF | Tensors | Notes |
|-------|------|------|---------|-------|
| cyberrealisticPony_v160 | 6.5G | 3.9G | 2,515 | Author: Cyberdelia |
| juggernautXL_ragnarokBy | 6.6G | 4.0G | 2,516 | |
| obsessiveCompulsive_v20 | 6.5G | 3.9G | 2,515 | |
| ponyDiffusionV6XL_v6StartWithThisOne | 6.5G | 3.9G | 2,515 | arch: `stable-diffusion-xl-v1-base` |
| ponyRealism_V22 | 6.6G | 4.0G | 2,515 | Merge: PonyRealism v2.1 + Volendir Cinematic v1.1R |
| realismIllustriousBy_v50FP16 | 6.5G | 3.9G | 2,515 | |
| spicyRealismNSFWMix_v30 | 6.5G | 3.9G | 2,515 | Triple merge |
| waiIllustriousSDXL_v160 | 6.5G | 3.9G | 2,515 | |

**Total checkpoints:** 11 models, ~95G (safetensors + GGUF)

## LoRAs

### SDXL / Illustrious

| LoRA | Size | GGUF | Dim | Alpha | Clip Skip | Title |
|------|------|------|-----|-------|-----------|-------|
| 70s_VPMS_V1-E20 | 218M | - | 32 | 32 | 2 | 70s Vintage Porn Magazine Style |
| Bimbo_Bomb_Girls_Pit_Style | 218M | 116M | 32 | 16 | 1 | Bimbo Bomb Girls Pit Style |
| Candy_Jab_Comix | 218M | 116M | 32 | 16 | 1 | Candy Jab Comix |
| Nellie_Jab_Comix-000009 | 218M | - | 32 | 16 | 1 | Nellie Jab Comix |
| RealisticAnimeIXL_v2 | 218M | 116M | 32 | 16 | 1 | RealisticAnimeIXL |
| Western_art_style (Melkor Mancin / Rizdraws) | 218M | - | 32 | 16 | 1 | Combined western art style |
| spumcostyle | 218M | 116M | 32 | 16 | 1 | spumcostyle |
| vitpitillust | 218M | 116M | 32 | 16 | - | vitpitillust |

### SD 1.5

| LoRA | Size | GGUF | Dim | Alpha | Clip Skip | Title |
|------|------|------|-----|-------|-----------|-------|
| BimboOne | 144M | 82M | 128 | 128 | 2 | BimboOne |
| Calm [MockAI - v1.0] | 37M | - | - | - | - | - |
| bimbo-fc-1.6a | 144M | - | - | - | - | - |
| bimbostyleTwo | 144M | 82M | 128 | 128 | 2 | bimbostyleTwo |

**Total LoRAs:** 12 models, ~3.4G (safetensors + GGUF)

## Character Presets

YAML files in `/models/characters/` with trigger words, positive/negative prompts per LoRA.

| LoRA | Presets |
|------|---------|
| 70s_VPMS | generic |
| BimboOne | candy_charms |
| Bimbo_Bomb_Girls_Pit_Style | blossom, bubbles, butter, generic |
| Calm_MockAI | generic |
| Candy_Jab_Comix | candy |
| Nellie_Jab_Comix | nellie |
| RealisticAnimeIXL | generic |
| Western_Melkor_Mancin | generic |
| bimbo-fc | generic |
| bimbostyleTwo | generic |
| spumcostyle | generic |
| vitpitillust | bimbo |

## Outputs

`/models/outputs/` — 167 generated images, 141M total
