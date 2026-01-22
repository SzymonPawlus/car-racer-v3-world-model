# Improving Car Racer
The model now is pretty good, but as it looks VAE and LSTM where trained on pretty terrible data.
In fact it were some of mine driving, which to be honest, wasn't really good.
And also there was some training data from the first version of the model that was also quite terrible.

Thus, the conclusion is that you could get better results by just doing this process:

- Get better training data using current model
- Get some worse data from human driving or some stupid/reckless model so that it can also see that world isn't all roses and roads
- Repeat the training process using all scripts here in exactly this order. YES It is that simple, but time consuming

Not sure what would be the final result, but I think it would be worth trying If I wanted to push the result

# Future Ventures
I would like to explore game `crafter2d`, but I have exam session in near near future so I can't but here are things I would like to explore:

- Follow Dreamer v2/v3 papers first
- Discrete Autoencoders
- RSSM (Recurrent state space models)
- Exploring potential connections between modulating connectoins between parts of models
- Hierarchical RL models
- Reccurent network training with evolution
- Inverse dreaming

They all look fun and I am really curious how much you can push on my old grumpy hardware.