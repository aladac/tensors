# Model Inventory

Location: `/models/`

## Checkpoints

### SD 1.5 (512x512)

| Model | CivitAI | Size | GGUF | Tensors | Notes |
|-------|---------|------|------|---------|-------|
| dreamshaper_8 | [DreamShaper](https://civitai.com/models/4384) | 2.0G | 1.7G | 1,131 | |
| epicrealism_naturalSinRC1VAE | [epiCRealism](https://civitai.com/models/25694) | 2.0G | 1.7G | 1,133 | Includes VAE |
| hassakuSD15_v13 | [Hassaku](https://civitai.com/models/2583) | 2.0G | 1.7G | 1,131 | |

### SDXL / Pony / Illustrious (1024x1024)

| Model | CivitAI | Size | GGUF | Tensors | Notes |
|-------|---------|------|------|---------|-------|
| cyberrealisticPony_v160 | [CyberRealistic Pony](https://civitai.com/models/443821) | 6.5G | 3.9G | 2,515 | Pony, Author: Cyberdelia |
| juggernautXL_ragnarokBy | [Juggernaut XL](https://civitai.com/models/133005) | 6.6G | 4.0G | 2,516 | SDXL 1.0 |
| obsessiveCompulsive_v20 | [Obsessive Compulsive Disorder](https://civitai.com/models/2223077) | 6.5G | 3.9G | 2,515 | Illustrious |
| ponyDiffusionV6XL_v6StartWithThisOne | [Pony Diffusion V6 XL](https://civitai.com/models/257749) | 6.5G | 3.9G | 2,515 | Pony |
| ponyRealism_V22 | - | 6.6G | 4.0G | 2,515 | Merge: PonyRealism v2.1 + Volendir Cinematic v1.1R |
| realismIllustriousBy_v50FP16 | [Realism Illustrious](https://civitai.com/models/974693) | 6.5G | 3.9G | 2,515 | Illustrious |
| spicyRealismNSFWMix_v30 | - | 6.5G | 3.9G | 2,515 | Triple merge |
| waiIllustriousSDXL_v160 | [WAI-illustrious-SDXL](https://civitai.com/models/827184) | 6.5G | 3.9G | 2,515 | Illustrious |

**Total checkpoints:** 11 models, ~95G (safetensors + GGUF)

## LoRAs

### SDXL / Illustrious

| LoRA | CivitAI | Size | GGUF | Dim | Alpha | Clip Skip |
|------|---------|------|------|-----|-------|-----------|
| 70s_VPMS_V1-E20 | [70s Vintage Porn Magazine Style](https://civitai.com/models/999258) | 218M | - | 32 | 32 | 2 |
| Bimbo_Bomb_Girls_Pit_Style | [Bimbo Bomb Girls (Pit Style)](https://civitai.com/models/1448347) | 218M | 116M | 32 | 16 | 1 |
| Candy_Jab_Comix | [Candy [Jab Comix]](https://civitai.com/models/2261458) | 218M | 116M | 32 | 16 | 1 |
| Nellie_Jab_Comix-000009 | [Nellie [Jab Comix]](https://civitai.com/models/2319798) | 218M | - | 32 | 16 | 1 |
| RealisticAnimeIXL_v2 | - | 218M | 116M | 32 | 16 | 1 |
| Western_art_style (Melkor Mancin / Rizdraws) | [Melkor Mancin / Rizdraws](https://civitai.com/models/1253845) | 218M | - | 32 | 16 | 1 |
| spumcostyle | [Spumco Cartoon Style](https://civitai.com/models/1825856) | 218M | 116M | 32 | 16 | 1 |
| vitpitillust | [Bimbo Art for WAI-NSFW-illustrious](https://civitai.com/models/1021102) | 218M | 116M | 32 | 16 | - |

### SD 1.5

| LoRA | CivitAI | Size | GGUF | Dim | Alpha | Clip Skip |
|------|---------|------|------|-----|-------|-----------|
| BimboOne | [bimbostyleOne](https://civitai.com/models/26340) | 144M | 82M | 128 | 128 | 2 |
| Calm [MockAI - v1.0] | [Calm - Style](https://civitai.com/models/174611) | 37M | - | - | - | - |
| bimbo-fc-1.6a | [Busty Petite Bimbo Doll -fc](https://civitai.com/models/64722) | 144M | - | - | - | - |
| bimbostyleTwo | [bimbostyleTwo](https://civitai.com/models/25795) | 144M | 82M | 128 | 128 | 2 |

**Total LoRAs:** 12 models, ~3.4G (safetensors + GGUF)

## Downloaded but Not Present

Models from `tsr dl` history that aren't in `/models/`:

| CivitAI ID | Name | Type | Base |
|------------|------|------|------|
| 4468 | [Counterfeit-V3.0](https://civitai.com/models/4468) | Checkpoint | SD 1.5 |
| 113483 | [Western Bimbos](https://civitai.com/models/113483) | LORA | SD 1.5 |
| 362745 | [CAT - Citron Styles](https://civitai.com/models/362745) | LORA | Illustrious |
| 2380160 | [Patausche Kivia (Sentenced to Be a Hero)](https://civitai.com/models/2380160) | LORA | Illustrious |

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
