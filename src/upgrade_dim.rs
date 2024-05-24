use burn::tensor::{backend::Backend, Shape, Tensor};

pub trait UpgradeDim<B: Backend, const N: usize> {
    fn upgrade_dim(self, to_shape: impl Into<Shape<N>>) -> Tensor<B, N>;
}

impl<B: Backend> UpgradeDim<B, 3> for Tensor<B, 2> {
    fn upgrade_dim(self, to_shape: impl Into<Shape<3>>) -> Tensor<B, 3> {
        let self_shape = to_shape.into();
        self.unsqueeze::<3>()
    }
}

#[test]
fn auto_burn_broadcast() {
    use burn::backend::Wgpu;
    type B = Wgpu;
    let device = <B as Backend>::Device::default();
    let a = Tensor::<B, 2>::ones([1, 2], &device);
    let b = Tensor::<B, 2>::ones([3, 2], &device);
    let c = a + b;
    assert_eq!(c.dims(), [3, 2]);
    assert!(c.equal_elem(2).all().into_scalar());
}

#[test]
fn unsqueeze_test() {
    use burn::backend::Wgpu;
    type B = Wgpu;
    let device = <B as Backend>::Device::default();
    let a_arr = [[1.0, 2.0], [3.0, 4.0]];
    let a = Tensor::<B, 2>::from_floats(a_arr, &device);
    let unsqueeze = a.unsqueeze::<3>();
    let test = Tensor::<B, 3>::from_floats([a_arr], &device);
    unsqueeze.to_data().assert_approx_eq(&test.to_data(), 1);
}
