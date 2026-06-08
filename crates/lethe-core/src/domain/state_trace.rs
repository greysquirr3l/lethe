use super::State;

#[derive(Debug, Clone, PartialEq)]
pub struct TraceFrame {
    pub tick: usize,
    pub state: State,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StateTrace {
    frames: Vec<TraceFrame>,
}

impl StateTrace {
    #[must_use]
    pub const fn new() -> Self {
        Self { frames: Vec::new() }
    }

    pub fn push(&mut self, tick: usize, state: State) {
        self.frames.push(TraceFrame { tick, state });
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.frames.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    #[must_use]
    pub fn frames(&self) -> &[TraceFrame] {
        &self.frames
    }
}
