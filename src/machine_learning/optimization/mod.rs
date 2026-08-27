mod adam;
mod gradient_descent;
mod momentum;
mod stochastic_gradient_descent;

pub use self::adam::Adam;
pub use self::gradient_descent::gradient_descent;
pub use self::stochastic_gradient_descent::stochastic_gradient_descent;
