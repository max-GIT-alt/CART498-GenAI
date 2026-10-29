# Postcards from My Jungle

**Live site:** https://max-GIT-alt.github.io/CART498-GenAI/A5

Written commentary:
Having very little time to mess with things, I gave my paid Claude Opus 5.5 the template, and enabled it to built the google-collab documents. I have run each of these documents and created 50 animals picture and 16 sound samples, all mixed to make the slideshow. I kept the generated "language" folder, despite tat not being needed, because I found the translation interesting in a weird way. The strategy, as explained, it is a simple data pull and diffusion train loop. For sound, I directly used a HF model. Having more time to train with a larger data set would enable quick improvements, but high quality diffusion these days are a matter of handling touch-ups, local complexity explosions and continuity. These are a matter of having a judge model, MCP and communication tools, integrated pipelines for rapid iteration, good GPU support and more. It has little to do with the model architecture or data amount/quality past a certain point, even if these do matter. Sound is something I have very little experience with. My personal project involves an emulator synth that translates json instructions into direct instrumental execution, but this is not industry standard. I quickly noticed that "better" is very difficult to explain in plain English, let alone trace to a specific harmony or wave overlap. 

AI generated:
*Postcards from My Jungle* is a lo-fi slideshow of fifty creatures that never existed. Each postcard pairs an image from a small diffusion model with a call from Stable Audio Open and a phrase in Ulvari, an invented animal language.

**Training strategy.** I trained an unconditional DDPM (`UNet2DModel` + `DDPMScheduler` from diffusers) from scratch in Google Colab on [3,000] images from [the "wild" class of AFHQv2]. A from-scratch model on a free GPU cannot learn real anatomy, so I treated its confusion as the point: half-learned fur, eyes and silhouettes blur into new species. I compared [1,000 images / 64 px / 30 epochs], [3,000 / 64 px / 60 epochs] and [3,000 / 128 px / 40 epochs]. [More data at 64 px gave the most coherent creatures; 128 px converged too slowly for my GPU time.] I sampled fifty images from the final checkpoint with 1,000 denoising steps.

**Sounds.** I wrote sixteen prompts, one per creature clan, describing body size, habitat and mood, and generated them with Stable Audio Open 1.0 at lengths from 2 to 20 seconds. A negative prompt removed music and speech; each clip was normalized, faded and converted to MP3.

**Language.** I gave [Claude] a structured prompt defining Ulvari's sounds, a sixteen-root lexicon, suffixes for "my," "our," past, future and questions, and a JSON format with phrase, word-by-word gloss and translation. These constraints keep the sixteen phrases consistent, so words like *ul* (sun) recur across creatures (see `language/`).

**Improvements.** Images would improve with a larger, cleaner dataset, longer training, an EMA of the weights, or fine-tuning a pretrained model instead of starting from scratch. Sounds would improve with more specific prompts, several takes per prompt to choose from, and light mixing so the calls share one jungle space.

**AI assistance.** [I used Claude (claude.ai, paid) to draft the website code, the notebooks and this README, and to generate the Ulvari phrases. I ran the training and generation and chose the final images and sounds myself.]
