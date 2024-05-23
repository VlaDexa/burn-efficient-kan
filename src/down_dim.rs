use burn::tensor::{backend::Backend, Bool, Float, Tensor, TensorKind};

use crate::upgrade_dim::UpgradeDim;

pub trait DownDimensionOps<B: Backend, const N1: usize, const N2: usize, T: TensorKind<B> = Float>
where
    Tensor<B, N2, T>: UpgradeDim<B, N1>,
{
    fn greater_equal_down(self, other: Tensor<B, N2, T>) -> Tensor<B, N1, Bool>;
    fn lower_down(self, other: Tensor<B, N2, T>) -> Tensor<B, N1, Bool>;
    fn sub_down(self, other: Tensor<B, N2, T>) -> Tensor<B, N1, T>;
    fn div_down(self, other: Tensor<B, N2, T>) -> Tensor<B, N1, T>;
}

impl<B: Backend> DownDimensionOps<B, 3, 2> for Tensor<B, 3> {
    fn greater_equal_down(self, other: Tensor<B, 2>) -> Tensor<B, 3, Bool> {
        let upgrade = other.upgrade_dim(self.shape());
        self.greater_equal(upgrade)
    }

    fn lower_down(self, other: Tensor<B, 2>) -> Tensor<B, 3, Bool> {
        let upgrade = other.upgrade_dim(self.shape());
        self.lower(upgrade)
    }

    fn sub_down(self, other: Tensor<B, 2>) -> Self {
        let upgrade = other.upgrade_dim(self.shape());
        self.sub(upgrade)
    }

    fn div_down(self, other: Tensor<B, 2>) -> Self {
        let upgrade = other.upgrade_dim(self.shape());
        self.div(upgrade)
    }
}

#[test]
fn greater_equal_down_test() {
    use burn::backend::Wgpu;
    type B = Wgpu;
    let device = <B as Backend>::Device::default();
    let a_arr = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let a = Tensor::<B, 3>::from_floats(a_arr, &device);
    let b_arr = [[2.0, 2.0], [2.0, 4.0]];
    let b = Tensor::<_, 2>::from_floats(b_arr, &device);
    let c = a.greater_equal_down(b);
    let test_res = a_arr.map(|a_two| {
        let mut res = a_two.map(|a| a.map(|_| false));
        for (i, a) in a_two.into_iter().enumerate() {
            for (j, a) in a.into_iter().enumerate() {
                res[j][i] = a >= b_arr[j][i];
            }
        }
        res
    });
    let test = Tensor::<B, 3, Bool>::from_bool(test_res.into(), &device);
    c.to_data().assert_approx_eq(&test.to_data(), 1);
}
