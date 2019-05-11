use land::prelude::*;

type Matrix = land::Matrix<f64>;
type Vector = land::Vector<f64>;

#[derive(Debug, Clone)]
pub struct NeuronLayer {
    weights: Matrix,
    basis: Vector,
    activation: Activation,
    loss: LossFunction,
}

#[derive(Debug, Copy, Clone)]
pub enum Activation {
    Sigmoid,
    RectifiedLinearUnit,
    Softplus,
}

#[derive(Debug, Copy, Clone)]
pub enum LossFunction {
    SquaredError,
    CrossEntropy,
}

#[derive(Debug, Clone)]
pub struct Propagation {
    /// Input pushed to the layer.
    input: Vector,

    /// Activated outputs from the layer.
    output: Vector,

    /// Adjustment vectors
    adjustment: Adjustment,

    learning_rate: f64,
}

#[derive(Debug, Clone)]
pub enum Adjustment {
    /// Adjust weights to approach a target output
    Target(Vector),

    /// Adjust weights based on output's impact on the error.
    Deltas(WeightedDeltas),
}

#[derive(Debug, Clone)]
pub struct WeightedDeltas(Vector);

impl NeuronLayer {
    /// Push inputs through the network and returns the activated output
    pub fn propagate(&self, input: &Vector) -> Vector {
        let net_output = &self.weights * input + &self.basis;
        let output = self.activation.apply(&net_output);
        output
    }

    /// Propagate errors backwards
    pub fn backpropagation(&mut self, propagation: &Propagation) -> WeightedDeltas {
        let deltas = match propagation.adjustment {
            Adjustment::Target(ref target) => {
                self.loss.derivative(&target, &propagation.output)
                    * self.activation.derivative(&propagation.output)
            }

            Adjustment::Deltas(WeightedDeltas(ref deltas)) => {
                deltas * self.activation.derivative(&propagation.output)
            }
        };

        let weighted_deltas = &self.weights.transpose() * &deltas;

        let corr = deltas.mul_transpose(&propagation.input);
        self.weights -= propagation.learning_rate * corr;

        WeightedDeltas(weighted_deltas)
    }
}

impl Activation {
    pub fn apply(&self, x: &Vector) -> Vector {
        match self {
            Activation::Sigmoid => ((-x).exp() + 1.0).recip(),

            Activation::RectifiedLinearUnit => x.clone().max(0.0),

            Activation::Softplus => (1.0 + x).ln(),
        }
    }

    pub fn derivative(&self, x: &Vector) -> Vector {
        match self {
            Activation::Sigmoid => {
                let sigmoid = self.apply(x);
                sigmoid.clone() * (1.0 - sigmoid)
            }

            Activation::RectifiedLinearUnit => x.clone().signum().max(0.0),

            Activation::Softplus => ((-x).exp() + 1.0).recip(),
        }
    }
}

impl LossFunction {
    pub fn error(&self, target: &Vector, actual: &Vector) -> Vector {
        match self {
            LossFunction::SquaredError => (target - actual).powi(2) / 2.0,

            LossFunction::CrossEntropy => unimplemented!(),
        }
    }

    pub fn derivative(&self, target: &Vector, actual: &Vector) -> Vector {
        match self {
            LossFunction::SquaredError => actual - target,

            LossFunction::CrossEntropy => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use land::mat;

    #[test]
    fn simple_propagation() {
        let layer = NeuronLayer {
            weights: mat![[1.0, 2.0]],
            basis: mat![0.0],
            activation: Activation::Sigmoid,
            loss: LossFunction::SquaredError,
        };

        let input = mat![0.4, -0.1];
        let output = layer.propagate(&input);

        assert_eq!(
            output,
            Activation::Sigmoid.apply(&mat![1.0 * 0.4 - 2.0 * 0.1])
        )
    }

    #[test]
    fn simple_backpropagation() {
        let mut layer = NeuronLayer {
            weights: mat![[1.0, 2.0]],
            basis: mat![0.0],
            activation: Activation::Sigmoid,
            loss: LossFunction::SquaredError,
        };

        let input = mat![0.4, -0.1];
        let expected = mat![0.3];

        for i in 0..1000 {
            let output = layer.propagate(&input);

            let propagation = Propagation { 
                input: input.clone(),
                output,
                adjustment: Adjustment::Target(expected.clone()),
                learning_rate: 1.0
            };

            layer.backpropagation(&propagation);
        }

        let output = layer.propagate(&input);

        assert!((output - expected).abs()[0] < 1e-3)
    }
}
