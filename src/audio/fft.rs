use std::collections::HashMap;

use ringbuf::{HeapRb, Rb};

pub struct FFTContainer {
    fft_buffers: HashMap<usize, HeapRb<f32>>,
}

const AUDIO_SAMPLE_SIZE: usize = 512;
const FFT_BUFFER_SIZE: usize = 8;
const TRANSIENT_THRESHOLD: f32 = 0.5;

impl FFTContainer {
    pub fn new() -> Self {
        let mut fft_buffers = HashMap::new();

        // allocate all buffers
        for i in 0..AUDIO_SAMPLE_SIZE {
            let mut rb = HeapRb::new(FFT_BUFFER_SIZE);

            // fill the buffer with zeros
            for _ in 0..FFT_BUFFER_SIZE {
                rb.push(0.0).unwrap();
            }

            fft_buffers.insert(i, rb);
        }

        FFTContainer { fft_buffers }
    }

    pub fn set_fft(&mut self, fft: Vec<f32>) {
        // if we don't have data, drop
        if fft.len() != AUDIO_SAMPLE_SIZE {
            return;
        }

        // copy the fft data into the buffers
        for (i, buffer) in self.fft_buffers.iter_mut() {
            // detect a transient
            let average_buffer = buffer.iter().sum::<f32>() / buffer.len() as f32;
            let diff = fft[*i] - average_buffer;
            if diff > TRANSIENT_THRESHOLD {
                buffer.clear();
            }

            buffer.push_overwrite(fft[*i]);
        }
    }

    pub fn read_fft(&self, smoothing_amount: usize) -> Vec<f32> {
        // TODO: Implement smoothing
        let _ = smoothing_amount;
        // Read the buffers into the fft, smoothed
        self.fft_buffers
            .iter()
            .map(|(_, buffer)| buffer.iter().sum::<f32>() / buffer.len() as f32)
            .collect()
    }
}
