# CAR RACER-v3 VMC Model
## Motivation
Project started when I saw some videos about "world models" in AI and I became really curious about the concept.
I started reading about it some papers and asking Gemini thinks about it. 
I had introduction to machine learning on my university and I understand the math behind it so I though that it might be nice to dive in.

## How it was done?
The project was mainly `vibe coded`. Normally I think of it as kinda shameful, but this is **PYTHON**
and also this isn't my field of expertise.
In beginning stages I used my Laptop that is quite average, but then to simulate the environment quicker I used 
a server machine with 72 CPU cores that my roommate built from old parts from `Allegro`.
The whole thing took less than 4 days to complete with help of Gemini Pro.

## What is CAR RACER-v3
If you don't know it is a 2D simple racing game, but with some non-trivial physics mechanics
that is great for benchmarking ML models, that need to have a vision and reflexive memory.
You can play it by yourself - it is not that easy, especially on keyboard.
### Scoring
It is important how score is calculated here.
The road consists of tiles with each tile being worth 1 point.
The whole race takes at most 1000 frames.
This gives us a total score for a race to be
``#tiles - 0.1 * #frames``
Basically results ~800 are considered good and results ~900 are considered fantastic

## My model
So I used VMC architecture (Vision - Memory - Controller). I will briefly
explain the structure

### Vision - Variation Autoencoder
Standard solution here - you take a lot of races (I just drove on most of them) and
you create an encoder that compresses the vision into latent vector of way smaller size that we will use in otehr models

### Memory - LSTM
Here basically comes the idea of dreaming that we take train a model on current state in latent space
and action that was taken and the output is the result of this action.
By this it learns how to dream into the future about how the future will unveil.
We will feed its hidden state which is like some kind of short term memory to the controller

### Controller - CMA-ES linear network
This is the most fascinating thing for me that after the whole preprocessing done by VM we are just left with a linear
model that we can just train evolutionarily. Here 72 CPU cores come in handy, because now I can simulate 72 environments in parallel
it yielded great results quickly!

## Results
Final model that I have created achieves `798.2` score on average from 10 different seeds.
This is a great result and you can see it driving quite well in `best_lap.mp4`
The steps to potentially improve the model and some other fun RL things are in `NEXT_STEPS.MD`
