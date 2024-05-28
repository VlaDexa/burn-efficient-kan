#![warn(clippy::nursery, clippy::pedantic)]
mod down_dim;
mod kaiming;
mod least_squares;
mod linear;
use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Tensor},
};
use linear::Linear;

#[derive(Debug, Module)]
pub struct Kan<B: Backend> {
    grid_size: u16,
    spline_order: u32,
    layer_one: Linear<B>,
    layer_two: Linear<B>,
}

#[derive(Config, Debug)]
pub struct KanOptions {
    pub layers_hidden: [u32; 3],
    #[config(default = 5)]
    pub grid_size: u16,
    #[config(default = 3)]
    pub spline_order: u32,
    #[config(default = 0.1)]
    pub scale_noise: f32,
    #[config(default = 1.0)]
    pub scale_base: f32,
    #[config(default = 1.0)]
    pub scale_spline: f32,
    #[config(default = 0.02)]
    pub grid_eps: f32,
    #[config(default = true)]
    pub enable_standalone_scale_spine: bool,
    #[config(default = -1)]
    pub grid_range_start: i32,
    #[config(default = 1)]
    pub grid_range_end: i32,
}

impl KanOptions {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Kan<B>
    where
        B::FloatElem: ndarray_linalg::Scalar + ndarray_linalg::Lapack,
    {
        Kan::new(self, device)
    }
}

impl<B: Backend> Kan<B> {
    pub fn new(options: &KanOptions, device: &B::Device) -> Self
    where
        B::FloatElem: ndarray_linalg::Scalar + ndarray_linalg::Lapack,
    {
        let layers_hidden = options.layers_hidden;
        let zip = [
            (layers_hidden[0], layers_hidden[1]),
            (layers_hidden[1], layers_hidden[2]),
        ];
        let [layer_one, layer_two] = zip.map(|(in_features, out_features)| {
            Linear::new(in_features, out_features, options, device)
        });
        Self {
            grid_size: options.grid_size,
            spline_order: options.spline_order,
            layer_one,
            layer_two,
        }
    }

    const fn layers(&self) -> [&Linear<B>; 2] {
        [&self.layer_one, &self.layer_two]
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut output = input;
        for layer in self.layers() {
            output = layer.forward(&output);
        }
        output
    }
}
