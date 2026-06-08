#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub activities: Vec<f64>,
    pub lambda_i: Vec<f64>,
}

impl State {
    #[must_use]
    pub const fn new(activities: Vec<f64>, lambda_i: Vec<f64>) -> Self {
        Self {
            activities,
            lambda_i,
        }
    }

    #[must_use]
    pub const fn cell_count(&self) -> usize {
        self.activities.len()
    }
}
