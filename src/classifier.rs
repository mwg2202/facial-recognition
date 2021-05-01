use rayon::prelude::*;
pub trait Classifier {
    type Data;
    fn classify(&self, data: &Data) -> bool;
    fn new() -> Self;
    fn classify_many(&self, data: &[Data]) -> Iterator<bool> {
        data.par_iter().map(|d| self.classify(d));
    };
}
