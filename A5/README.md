# Postcards from My Jungle

**Live site:** https://YOUR-USERNAME.github.io/postcards-from-my-jungle/

> Draft — replace every [bracketed] part with what you actually did and observed, then check the word count (250–350).

*Postcards from My Jungle* is a lo-fi slideshow of fifty creatures that never existed. Each postcard pairs an image from a small diffusion model with a call from Stable Audio Open and a phrase in Ulvari, an invented animal language.

**Training strategy.** I trained an unconditional DDPM (`UNet2DModel` + `DDPMScheduler` from diffusers) from scratch in Google Colab on [3,000] images from [the "wild" class of AFHQv2]. A from-scratch model on a free GPU cannot learn real anatomy, so I treated its confusion as the point: half-learned fur, eyes and silhouettes blur into new species. I compared [1,000 images / 64 px / 30 epochs], [3,000 / 64 px / 60 epochs] and [3,000 / 128 px / 40 epochs]. [More data at 64 px gave the most coherent creatures; 128 px converged too slowly for my GPU time.] I sampled fifty images from the final checkpoint with 1,000 denoising steps.

**Sounds.** I wrote sixteen prompts, one per creature clan, describing body size, habitat and mood, and generated them with Stable Audio Open 1.0 at lengths from 2 to 20 seconds. A negative prompt removed music and speech; each clip was normalized, faded and converted to MP3.

**Language.** I gave [Claude] a structured prompt defining Ulvari's sounds, a sixteen-root lexicon, suffixes for "my," "our," past, future and questions, and a JSON format with phrase, word-by-word gloss and translation. These constraints keep the sixteen phrases consistent, so words like *ul* (sun) recur across creatures (see `language/`).

**Improvements.** Images would improve with a larger, cleaner dataset, longer training, an EMA of the weights, or fine-tuning a pretrained model instead of starting from scratch. Sounds would improve with more specific prompts, several takes per prompt to choose from, and light mixing so the calls share one jungle space.

**AI assistance.** [I used Claude (claude.ai, free / paid — say which) to draft the website code, the notebooks and this README, and to generate the Ulvari phrases. I ran the training and generation and chose the final images and sounds myself.]
