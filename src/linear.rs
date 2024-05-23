use crate::{
    down_dim::DownDimensionOps,
    kaiming::{kaiming_uniform_param, KaimingMode, KaimingNonlinearity},
    least_squares::LeastSquares,
    KanOptions,
};
use burn::module::{Module, Param, ParamId};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use ndarray_linalg::Scalar;

#[derive(Debug, Module)]
pub struct Linear<B: Backend> {
    in_features: u32,
    out_features: u32,
    grid_size: u16,
    spline_order: u32,
    grid: Tensor<B, 2>,
    base_weight: Param<Tensor<B, 2>>,
    spline_weight: Param<Tensor<B, 3>>,
    spline_scaler: Option<Param<Tensor<B, 2>>>,
    scale_noise: f32,
    scale_base: f32,
    scale_spline: f32,
    grid_eps: f32,
    enable_standalone_scale_spine: bool,
}

impl<B: Backend> Linear<B> {
    pub(crate) fn new(
        in_features: u32,
        out_features: u32,
        options: &KanOptions,
        device: &B::Device,
    ) -> Self
    where
        B::FloatElem: ndarray_linalg::Scalar + ndarray_linalg::Lapack,
    {
        let KanOptions {
            grid_size,
            spline_order,
            grid_range_start,
            grid_range_end,
            scale_noise,
            scale_base,
            enable_standalone_scale_spine,
            grid_eps,
            scale_spline,
            ..
        } = *options;
        let h = (grid_range_end - grid_range_start) as f32 / f32::from(grid_size);
        let grid = (Tensor::<B, 1, Int>::arange(
            (-i64::from(spline_order))..(i64::from(grid_size) + i64::from(spline_order) + 1),
            device,
        )
        .float()
            * h
            + grid_range_start)
            .expand([(in_features as i32), -1]);
        let base_weight = kaiming_uniform_param(
            &[(in_features as usize), (out_features as usize)],
            5.0.sqrt() * f64::from(scale_base),
            KaimingMode::default(),
            KaimingNonlinearity::default(),
            device,
        )
        .set_require_grad(true);
        let noise = ((Tensor::<B, 3>::random(
            [
                usize::from(grid_size + 1),
                (in_features as usize),
                (out_features as usize),
            ],
            burn::tensor::Distribution::Default,
            device,
        ) - 1 / 2)
            * scale_noise
            / u32::from(grid_size))
        .set_require_grad(false);
        let grid_clone = grid.clone();
        let grid_trans = grid_clone.clone().transpose();
        let spline_weight = Param::uninitialized(
            ParamId::new(),
            move |_device, grad| {
                let mult = if enable_standalone_scale_spine {
                    1.0
                } else {
                    scale_spline
                };
                Self::curve2coeff(
                    grid_trans.clone().slice([
                        spline_order as usize..(grid_trans.dims()[0] - spline_order as usize),
                        0..grid_trans.dims()[1],
                    ]),
                    noise.clone(),
                    in_features,
                    out_features,
                    grid_size,
                    spline_order,
                    &grid_clone,
                )
                .mul_scalar(mult)
                .set_require_grad(grad)
            },
            device.clone(),
            false,
        );
        let spline_scaler = enable_standalone_scale_spine.then(|| {
            kaiming_uniform_param(
                &[(out_features as usize), (in_features as usize)],
                5.0.sqrt() * f64::from(scale_spline),
                KaimingMode::default(),
                KaimingNonlinearity::default(),
                device,
            )
            .set_require_grad(false)
        });
        Self {
            in_features,
            out_features,
            grid_size,
            spline_order,
            grid,
            base_weight,
            spline_weight,
            spline_scaler,
            scale_noise,
            scale_base,
            scale_spline,
            grid_eps,
            enable_standalone_scale_spine,
        }
    }

    fn curve2coeff(
        x: Tensor<B, 2>,
        y: Tensor<B, 3>,
        in_features: u32,
        out_features: u32,
        grid_size: u16,
        spline_order: u32,
        grid: &Tensor<B, 2>,
    ) -> Tensor<B, 3>
    where
        B::FloatElem: ndarray_linalg::Scalar + ndarray_linalg::Lapack,
    {
        let batch_size = x.dims()[0];
        assert_eq!(x.dims().len(), 2);
        assert_eq!(x.dims(), [batch_size, in_features as usize]);
        assert_eq!(
            y.dims(),
            [x.dims()[0], (in_features as usize), (out_features as usize)]
        );
        let device = x.device();
        let a = Self::b_splines(x, grid, spline_order, in_features, grid_size).swap_dims(0, 1);
        assert_eq!(
            a.dims(),
            [
                (in_features as usize),
                batch_size,
                (u32::from(grid_size) + spline_order) as usize
            ]
        );
        let b = y.swap_dims(0, 1);
        assert_eq!(
            b.dims(),
            [(in_features as usize), batch_size, (out_features as usize)]
        );
        let mut result: Tensor<B, 3> = Tensor::from_data(a.burn_least_squares(b).unwrap(), &device);
        assert_eq!(
            result.dims(),
            [
                (in_features as usize),
                (u32::from(grid_size) + spline_order) as usize,
                (out_features as usize),
            ]
        );
        result = result.permute([2, 0, 1]);
        assert_eq!(
            result.dims(),
            [
                (out_features as usize),
                (in_features as usize),
                (u32::from(grid_size) + spline_order) as usize
            ]
        );
        result
    }

    fn b_splines(
        x: Tensor<B, 2>,
        grid: &Tensor<B, 2>,
        spline_order: u32,
        in_features: u32,
        grid_size: u16,
    ) -> Tensor<B, 3> {
        assert_eq!(x.dims().len(), 2);
        assert_eq!(x.dims()[1], (in_features as usize));
        let x = x.unsqueeze_dim::<3>(2);
        let grid_dims = grid.dims();
        let mut bases = x
            .clone()
            .greater_equal_down(grid.clone().slice([0..grid_dims[0], 0..grid_dims[1] - 1]))
            .equal(
                x.clone()
                    .lower_down(grid.clone().slice([0..grid_dims[0], 0..grid_dims[1] - 1])),
            )
            .float();
        let minus_one = x.zeros_like().sub(x.ones_like());
        for k in 1..=(spline_order as usize) {
            let grid_minus_k_plus_one = grid
                .clone()
                .slice([0..grid_dims[0], 0..(grid_dims[1] - (k + 1))]);
            let grid_k_plus_one = grid.clone().slice([0..grid_dims[0], k + 1..grid_dims[1]]);
            let bases_dim = bases.dims();
            let bases_slice = bases.slice([0..bases_dim[0], 0..bases_dim[1], 1..bases_dim[2]]);
            bases = x
                .clone()
                .sub_down(grid_minus_k_plus_one.clone())
                .div_down(
                    grid.clone()
                        .slice([0..grid_dims[0], k..grid_dims[1] - 1])
                        .sub(grid_minus_k_plus_one),
                )
                .mul(bases_slice.clone())
                .add(
                    x.clone()
                        .sub_down(grid_k_plus_one.clone())
                        .mul(minus_one.clone())
                        .div_down(
                            grid_k_plus_one
                                .sub(grid.clone().slice([0..grid_dims[0], 1..grid_dims[1] - k])),
                        )
                        .mul(bases_slice),
                );
        }
        assert_eq!(
            bases.dims(),
            [
                x.dims()[0],
                (in_features as usize),
                (u32::from(grid_size) + spline_order) as usize
            ]
        );
        bases
    }

    fn scaled_spline_weight(&self) -> Tensor<B, 3> {
        let spline_weight = self.spline_weight.val();
        if let Some(spline_scaler) = &self.spline_scaler {
            spline_weight.mul(
                spline_scaler
                    .val()
                    .unsqueeze_dim::<3>(spline_scaler.dims().len()),
            )
        } else {
            spline_weight
        }
    }

    pub(crate) fn forward(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        assert_eq!(x.dims()[1], self.in_features as usize);
        let base_output = burn::nn::Linear {
            weight: self.base_weight.clone(),
            bias: None,
        }
        .forward(burn::tensor::activation::silu(x.clone()));
        let spline_output_input = Self::b_splines(
            x.clone(),
            &self.grid,
            self.spline_order,
            self.in_features,
            self.grid_size,
        );
        let spline_output_input = spline_output_input.reshape([x.dims()[0] as i32, -1]);

        let spline_output = burn::nn::Linear {
            weight: Param::from_tensor(
                self.scaled_spline_weight()
                    .reshape([self.out_features as i32, -1])
                    .transpose(),
            ),
            bias: None,
        }
        .forward(spline_output_input);

        base_output.add(spline_output)
    }
}
