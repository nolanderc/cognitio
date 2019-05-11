
use crate::neuron::NeuronLayer;

pub struct Model {
    stages: Vec<Stage>
}

struct Stage {
    transforms: Vec<Transform>
}

enum Transform {
    NeuronLayer(NeuronLayer)
}

