use burn::tensor::{backend::Backend, Data, Shape, Tensor};
use ndarray_linalg::LeastSquaresResult;

pub trait LeastSquares {
    type B;
    type Out;

    fn burn_least_squares(self, b: Self::B) -> Self::Out;
}

impl<B: Backend> LeastSquares for Tensor<B, 2>
where
    B::FloatElem: ndarray_linalg::Scalar + ndarray_linalg::Lapack,
{
    type B = Self;
    type Out = ndarray_linalg::error::Result<Data<B::FloatElem, 2>>;

    fn burn_least_squares(self, b: Self::B) -> Self::Out {
        use ndarray_linalg::LeastSquaresSvd;
        // Convert to ndarray
        let a =
            ndarray::Array::from_shape_vec((self.dims()[0], self.dims()[1]), self.to_data().value)
                .unwrap();
        let b =
            ndarray::Array::from_shape_vec((b.dims()[0], b.dims()[1]), b.to_data().value).unwrap();
        // Perform least squares
        a.least_squares(&b)
            .map(|LeastSquaresResult { solution, .. }| {
                let shape = solution.raw_dim();
                Data::new(solution.into_raw_vec(), [shape[0], shape[1]].into())
            })
    }
}

impl<B: Backend> LeastSquares for Tensor<B, 3>
where
    B::FloatElem: ndarray_linalg::Scalar + ndarray_linalg::Lapack,
{
    type B = Self;
    type Out = ndarray_linalg::error::Result<Data<B::FloatElem, 3>>;

    fn burn_least_squares(self, b: Self::B) -> Self::Out {
        let b_dims = b.dims();
        let self_dims = self.dims();
        assert_eq!(self_dims[0..=1], b_dims[0..=1]);
        let (out_features, in_features, grid_spline_sum) = (b_dims[2], b_dims[0], self_dims[2]);
        assert!(
            b_dims[0] == self_dims[0] || is_singleton(&b),
            "The size of tensor a ({}) must match the size of tensor b ({}) at non-singleton dimension 0", b_dims[0], self_dims[0]
        );
        let mut result = Vec::with_capacity(b_dims[0] * b_dims[2] * self_dims[2]);
        match b_dims[0] {
            1 => {
                let b = b.squeeze::<2>(0);
                for a in self.iter_dim(0) {
                    let a = a.squeeze::<2>(0);
                    match a.burn_least_squares(b.clone()) {
                        Ok(solution) => result.extend(solution.value),
                        Err(e) => return Err(e),
                    }
                }
            }
            _ => {
                for (a, b) in self.iter_dim(0).zip(b.iter_dim(0)) {
                    let a = a.squeeze::<2>(0);
                    let b = b.squeeze::<2>(0);
                    match a.burn_least_squares(b.clone()) {
                        Ok(solution) => result.extend(solution.value),
                        Err(e) => return Err(e),
                    }
                }
            }
        };
        Ok(Data::new(
            result,
            Shape::from([in_features, grid_spline_sum, out_features]),
        ))
    }
}

/// Checks if a tensor is a singleton
/// # Examples
/// ```ignore
/// let singleton = Tensor::<B, 2>::zeros([1,2], &device);
/// assert!(is_singleton(singleton));
///
/// let not_singleton = Tensor::<B, 2>::zeros([2,2], &device);
/// assert!(!is_singleton(not_singleton));
/// ```
fn is_singleton<
    B: Backend,
    const D: usize,
    K: burn::tensor::TensorKind<B> + burn::tensor::BasicOps<B>,
>(
    tensor: &Tensor<B, D, K>,
) -> bool {
    tensor.dims()[0] == 1
}

#[test]
fn is_singleton_test() {
    use burn::backend::Wgpu;
    type B = Wgpu;
    let device = <B as Backend>::Device::default();
    let singleton: Tensor<B, 2> = Tensor::zeros([1, 2], &device);
    assert!(is_singleton(&singleton));
    let not_singleton: Tensor<B, 2> = Tensor::zeros([2, 2], &device);
    assert!(!is_singleton(&not_singleton));
}

#[test]
fn ndarray_torch_compat() {
    use ndarray::array;
    use ndarray_linalg::LeastSquaresSvd;
    // Create a 2x6
    let a = array![[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]];
    // Create a 2x3
    let b = array![[1., 2., 3.], [4., 5., 6.]];
    // Perform least squares
    let res = a.least_squares(&b);
    assert!(res.is_ok());
}
