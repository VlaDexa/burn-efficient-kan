use burn::{
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use burn_efficient_kan::{Kan as EfficientKan, KanOptions};

use crate::data::MnistBatch;

#[derive(Module, Debug)]
pub struct Kan<B: Backend> {
    kan: EfficientKan<B>,
}

impl<B: Backend> Kan<B> {
    pub fn new(options: &KanOptions, device: &B::Device) -> Self
    where
        B::FloatElem: ndarray_linalg::Scalar + ndarray_linalg::Lapack,
    {
        let kan = EfficientKan::new(options, device);

        Self { kan }
    }
}

impl<B: Backend> Kan<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();
        let x = input.reshape([batch_size, height * width]);
        
        self.kan.forward(x)
    }

    pub fn forward_classification(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Kan<B> {
    fn step(&self, item: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Kan<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch)
    }
}
