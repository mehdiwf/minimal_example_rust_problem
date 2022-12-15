#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(unused)]
#![allow(dead_code)]

use ndarray::prelude::*;
use rayon::prelude::*;

// -----------------------------------------
// STRUCTS AND METHOD DEFINITION
// -----------------------------------------

#[derive(Debug, PartialEq)]
struct vec2D {pub x: f64, pub y: f64}

#[derive(Debug, PartialEq)]
 struct ScalarField2D {
     s: Array2<f64>,
}    

#[derive(Debug, PartialEq)]
 struct VectorField2D {
     x: ScalarField2D,
     y: ScalarField2D
}

impl ScalarField2D {
     fn new(x_size: usize, y_size: usize) -> ScalarField2D {
        return ScalarField2D {
        s: Array::<f64, Ix2>::zeros((y_size, x_size).f())};}

     fn get_pos(&self, x: usize, y: usize) -> f64
    {return self.s[[y,x]];}

    // unsafe here?
    fn set_pos(&mut self, x: usize, y: usize, f: f64)
    {self.s[[y,x]] = f;}
}

impl VectorField2D {

     fn new(nrow: usize, ncol: usize) -> VectorField2D {
        return VectorField2D {
            x: ScalarField2D::new(nrow, ncol),
            y: ScalarField2D::new(nrow, ncol)}}

     fn get_pos(&self, x: usize, y: usize) -> vec2D
    {let vec_at_pos = vec2D {
        x: self.x.get_pos(x, y),
        y: self.y.get_pos(x, y)};
     return vec_at_pos;}

    // unsafe here?
    fn set_pos(&mut self, x: usize, y: usize, vec: &vec2D)
    {self.x.set_pos(x, y, vec.x);
     self.y.set_pos(x, y, vec.y);}

}

enum DerivDirection {
    X_axis,
    Y_axis}

// -----------------------------------------
// FUNCTIONS DEFINITION
// -----------------------------------------

fn partial_deriv(a: &ScalarField2D,
                 x: i32, y: i32,
                 direction: DerivDirection,
                 x_max: i32, y_max: i32) -> f64
{
    // i+1 with Periodic Boundaries
    let ip = ((x+1) % x_max) as usize;
    // i-1 with Periodic Boundaries
    let im = ((x - 1 + x_max) % x_max) as usize;
    // j+1 with Periodic Boundaries
    let jp = ((y+1) % y_max) as usize;
    // j-1 with Periodic Boundaries
    let jm = ((y - 1 + y_max) % y_max) as usize;
    let (i, j) = (x as usize, y as usize);

    match direction
        {
            DerivDirection::X_axis => {
                let derivative =
                    (a.get_pos(ip, j) - a.get_pos(im, j))/(2.);
                return derivative;
            },
            DerivDirection::Y_axis => {
                let derivative =
                    (a.get_pos(i, jp) - a.get_pos(i, jm))/(2.);
                return derivative;
            }}}

// computes the gradient of a scalar field at a given position
fn grad_scalar(scalar_field: &ScalarField2D,
               x: i32, y: i32,
               x_max: i32, y_max: i32) -> vec2D
{
    let grad = vec2D {
        x: partial_deriv(&scalar_field, x, y,
                         DerivDirection::X_axis,
                         x_max, y_max),
        y: partial_deriv(&scalar_field, x, y,
                         DerivDirection::Y_axis,
                         x_max, y_max)};
        
    return grad;
}

// -----------------------------------------
// MAIN
// -----------------------------------------

fn main() {

    let (x_max, y_max) = (2usize, 50usize);
    let (x_maxi32, y_maxi32) = (x_max as i32, y_max as i32);

    let mut GD_grad_rho = VectorField2D::new(x_max, y_max);
    let mut GD_rho = ScalarField2D::new(x_max, y_max);    

    let x_iterator = (0..x_max).into_par_iter();
    x_iterator.map(|xi| {
        let y_iterator = (0..y_max).into_par_iter();
        y_iterator.map(|yi| {

            // unsafe here?
            GD_grad_rho
                .set_pos(xi, yi,
                         &grad_scalar(&GD_rho,
                                      xi as i32, yi as i32,
                                      x_maxi32, y_maxi32));
            
        });});
}
