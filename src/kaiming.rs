use burn::{
    module::Param,
    nn::Initializer,
    tensor::{backend::Backend, Tensor},
};

#[derive(Default, Clone, Copy)]
pub enum KaimingMode {
    #[default]
    FanIn,
    FanOut,
}

#[derive(Default, Clone, Copy)]
pub enum KaimingNonlinearity {
    #[default]
    LeakyRelu,
    Relu,
}

fn calculate_gain(nonlinearity: KaimingNonlinearity, param: Option<f64>) -> f64 {
    match nonlinearity {
        KaimingNonlinearity::LeakyRelu => {
            let negative_slope = param.unwrap_or(0.01);
            (2.0 / negative_slope.mul_add(negative_slope, 1.0)).sqrt()
        }
        KaimingNonlinearity::Relu => core::f64::consts::SQRT_2,
    }
}

pub fn kaiming_uniform_param<B: Backend, const SIZE: usize>(
    dims: &[usize; SIZE],
    a: f64,
    mode: KaimingMode,
    nonlinearity: KaimingNonlinearity,
    device: &B::Device,
) -> Param<Tensor<B, SIZE>> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(dims);
    let gain = calculate_gain(nonlinearity, Some(a));
    Initializer::KaimingUniform {
        gain,
        fan_out_only: matches!(mode, KaimingMode::FanOut),
    }
    .init_with(*dims, Some(fan_in), Some(fan_out), device)
}

fn calculate_fan_in_fan_out<const SIZE: usize>(tensor: &[usize; SIZE]) -> (usize, usize) {
    assert!(
        SIZE >= 2,
        "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
    );

    let num_input_fmaps = tensor[1];
    let num_output_fmaps = tensor[0];
    let mut receptive_field_size = 1;
    if SIZE > 2 {
        for s in &tensor[2..] {
            receptive_field_size *= s;
        }
    }
    let fan_in = num_input_fmaps * receptive_field_size;
    let fan_out = num_output_fmaps * receptive_field_size;

    (fan_in, fan_out)
}
