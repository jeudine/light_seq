use core::f32::consts::PI;
use cpal::{
    platform::Stream,
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, Sample,
};
use crossterm::{cursor, terminal, ExecutableCommand};
use promptly::prompt_default;
use rand::prelude::*;
use rand::seq::SliceRandom;
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use rppal::gpio::Gpio;
use std::error::Error;
use std::io::{stdout, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{collections::VecDeque, fs::File};

pub const NB_AUDIO_CHANNELS: usize = 3;
const CHUNCK_SIZE: usize = 2048;
const STAT_WINDOW_DURATION: usize = 10; // In seconds
const SWAP_TIME: [f32; 2] = [30.0, 30.0];

#[derive(Copy, Clone)]
pub struct Data {
    pub gain: [f32; NB_AUDIO_CHANNELS],
    _offset: f32,
}

impl Data {
    pub fn new() -> Data {
        Data {
            gain: [0.0; NB_AUDIO_CHANNELS],
            _offset: 0.0,
        }
    }
}

struct Buffer {
    input: Vec<f32>,
    output: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    window: Vec<f32>,
    pos: usize,
    len: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    mean: Vec<f32>,
    var: Vec<f32>,
    count: u64,
    index_limits: Vec<usize>,
    stat_window: Vec<VecDeque<f32>>,
    stat_window_size: usize,
}
fn shuffle(rng: &mut ThreadRng) -> [u32; 3] {
    let mut res = [0, 1, 2];
    res.shuffle(rng);
    res
}
fn main() {
    let (d, _s) = init().unwrap();
    let mut rng = rand::thread_rng();

    let gpio = Gpio::new().unwrap();

    let mut pins = [
        gpio.get(0).unwrap().into_output(),
        gpio.get(1).unwrap().into_output(),
        gpio.get(2).unwrap().into_output(),
    ];

    let mut l_alloc = shuffle(&mut rng);

    let now = Instant::now();
    let time = now.elapsed().as_secs_f32();
    let mut shuffle_time = SWAP_TIME[0] + rng.gen::<f32>() * SWAP_TIME[1] + time;

    loop {
        let time = now.elapsed().as_secs_f32();
        if time > shuffle_time {
            l_alloc = shuffle(&mut rng);
            shuffle_time = SWAP_TIME[0] + rng.gen::<f32>() * SWAP_TIME[1] + time;
        }
        let d = d.lock().unwrap();
        let gain = d.gain;
        for i in 1..3 {
            if gain[i] > 2.0 {
                pins[l_alloc[i] as usize].set_high();
            } else if gain[i] <= 1.0 {
                pins[l_alloc[i] as usize].set_low();
            }
        }
    }
}
fn init() -> Result<(Arc<Mutex<Data>>, Stream), Box<dyn Error>> {
    let min_freq = 20;
    let max_freq = 20000;
    let host = cpal::default_host();
    let devices: Vec<_> = host.input_devices()?.collect();

    for (i, d) in devices.iter().enumerate() {
        println!("[DEVICE {}] {}", i, d.name()?);
    }

    let device_id: usize = prompt_default("Select Device Id", 0)?;

    let device = &devices[device_id];

    let config = device.default_input_config()?;

    println!("[DEFAULT AUDIO CONFIG] {:?}", config);

    let mut real_planner = RealFftPlanner::<f32>::new();
    let r2c = real_planner.plan_fft_forward(CHUNCK_SIZE);
    let input = r2c.make_input_vec();
    let output = r2c.make_output_vec();
    let scratch = r2c.make_scratch_vec();
    let stat_window = vec![VecDeque::new(); NB_AUDIO_CHANNELS];
    let hanning_window = (0..input.len())
        .map(|i| 0.5 * (1.0 - ((2.0 * PI * i as f32) / (input.len() - 1) as f32).cos()))
        .collect();

    let sample_rate = config.sample_rate().0;
    let mut buffer = Buffer {
        input,
        output,
        scratch,
        len: CHUNCK_SIZE,
        pos: 0,
        r2c,
        mean: vec![0.0; NB_AUDIO_CHANNELS],
        var: vec![0.0; NB_AUDIO_CHANNELS],
        count: 0,
        window: hanning_window,
        index_limits: calculate_channel_index(
            min_freq,
            max_freq,
            NB_AUDIO_CHANNELS as u32,
            sample_rate,
            CHUNCK_SIZE,
        ),
        stat_window,
        stat_window_size: sample_rate as usize * STAT_WINDOW_DURATION / CHUNCK_SIZE,
    };

    let mut profile = if cfg!(feature = "profile") {
        let f = File::create("profile.tsv").unwrap();
        let now = Instant::now();
        Some((f, now))
    } else {
        None
    };

    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {}", err);
    };

    let audio_data = Data {
        gain: [0.0; NB_AUDIO_CHANNELS],
        _offset: 0.0,
    };
    let audio_data_arc = Arc::new(Mutex::new(audio_data));
    let audio_data_arc1 = audio_data_arc.clone();

    let stream = match config.sample_format() {
        cpal::SampleFormat::I8 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                handle_input::<i8>(data, &mut buffer, &audio_data_arc1, &mut profile)
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                handle_input::<i16>(data, &mut buffer, &audio_data_arc1, &mut profile)
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I32 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                handle_input::<i32>(data, &mut buffer, &audio_data_arc1, &mut profile)
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                handle_input::<f32>(data, &mut buffer, &audio_data_arc1, &mut profile)
            },
            err_fn,
            None,
        )?,
        _ => return Err(Box::from("Unsupported sample format")),
    };

    for _ in 0..NB_AUDIO_CHANNELS {
        println!("");
    }

    stream.play()?;
    Ok((audio_data_arc, stream))
}

fn calculate_channel_index(
    min_freq: u32,
    max_freq: u32,
    nb_channels: u32,
    sample_rate: u32,
    chunck_size: usize,
) -> Vec<usize> {
    let nb_octaves = (max_freq as f32 / min_freq as f32).log2();
    let nb_octaves_per_channel = nb_octaves / nb_channels as f32;
    let index_limits = (0..nb_channels + 1)
        .map(|i| {
            min_freq as usize * 2_f32.powf(nb_octaves_per_channel * i as f32) as usize * chunck_size
                / sample_rate as usize
        })
        .collect();

    index_limits
}

fn handle_input<T>(
    input: &[T],
    buffer: &mut Buffer,
    audio_data: &Arc<Mutex<Data>>,
    profile: &mut Option<(File, Instant)>,
) where
    T: Sample,
    f32: FromSample<T>,
{
    // every 2 because stereo
    for &sample in input.iter().step_by(2) {
        let pos = buffer.pos;
        // apply window
        buffer.input[pos] = f32::from_sample(sample) * buffer.window[pos];
        buffer.pos = pos + 1;
        if buffer.pos == buffer.len {
            buffer.pos = 0;
            buffer.count += 1;
            buffer
                .r2c
                .process_with_scratch(&mut buffer.input, &mut buffer.output, &mut buffer.scratch)
                .unwrap();

            // compute levels
            let levels: Vec<_> = (0..NB_AUDIO_CHANNELS)
                .map(|x| {
                    (buffer.index_limits[x] + 1..buffer.index_limits[x + 1])
                        .fold(0.0, |acc, i| acc + buffer.output[i].norm())
                })
                .collect();

            // update mean, sd and stat_window
            let tmp_inv = 1.0 / (buffer.stat_window_size) as f32;

            // Initialization
            if buffer.count <= buffer.stat_window_size as u64 {
                for i in 0..NB_AUDIO_CHANNELS {
                    buffer.stat_window[i].push_front(levels[i]);
                    buffer.mean[i] += tmp_inv * levels[i];
                    buffer.var[i] += tmp_inv * levels[i].powi(2);
                    if buffer.count == buffer.stat_window_size as u64 {
                        buffer.var[i] -= buffer.mean[i].powi(2);
                    }
                }
            } else {
                for i in 0..NB_AUDIO_CHANNELS {
                    let last_val = buffer.stat_window[i].pop_back().unwrap();
                    buffer.stat_window[i].push_front(levels[i]);

                    let cur_mean = buffer.mean[i];

                    buffer.mean[i] = cur_mean + tmp_inv * (levels[i] - last_val);
                    buffer.var[i] = buffer.var[i]
                        + tmp_inv * (levels[i].powi(2) - last_val.powi(2))
                        + (cur_mean.powi(2) - buffer.mean[i].powi(2));

                    if buffer.var[i] < 0.0 {
                        buffer.var[i] = 0.0;
                    }
                }
            }

            //check if there is at least one value above the threshold
            let threshold = 5.0;
            let mut above = false;
            for x in &buffer.output {
                if x.norm() > threshold {
                    above = true;
                    break;
                }
            }

            let mut gain = [f32::MIN; NB_AUDIO_CHANNELS];
            if above {
                for i in 0..NB_AUDIO_CHANNELS {
                    gain[i] = (levels[i] - buffer.mean[i]) / buffer.var[i].sqrt();
                }
            }

            let mut stdout = stdout();
            stdout.execute(cursor::MoveUp(gain.len() as u16)).unwrap();
            stdout
                .execute(terminal::Clear(terminal::ClearType::FromCursorDown))
                .unwrap();
            for (i, g) in gain.iter().enumerate() {
                if *g == f32::MIN {
                    writeln!(stdout, "audio_channel[{}]: -INF", i).unwrap();
                } else {
                    writeln!(stdout, "audio_channel[{}]: {}", i, g).unwrap();
                }
            }

            if let Some((ref mut f, n)) = profile {
                write!(f, "{}", n.elapsed().as_secs_f32()).unwrap();
                for g in gain {
                    write!(f, "\t{}", g).unwrap();
                }
                writeln!(f, "").unwrap();
            }

            let mut audio_data = audio_data.lock().unwrap();
            audio_data.gain = gain;
            return;
        }
    }
}