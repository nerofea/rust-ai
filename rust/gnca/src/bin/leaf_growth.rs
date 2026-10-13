use rayon::prelude::*;
use std::time::Instant;
const N: usize = 24;
const STEPS: usize = 12;
const K: usize = 9;
const DT: f32 = 0.2;
const RATE: f32 = 0.75;
type W = [[f32; K]; 2];
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cell {
    growth: f32,
    vein: f32,
    x: f32,
    y: f32,
}
fn target(x: f32, y: f32) -> [f32; 2] {
    let r = y / 0.87;
    let half = 0.08 + 0.39 * (1. - r * r).max(0.).sqrt();
    let blade = if y.abs() < 0.87 && (x / half).powi(2) + (y / 0.87).powi(2) < 1. {
        1.
    } else {
        0.
    };
    let vein = if blade > 0. && x.abs() < 0.07 { 1. } else { 0. };
    [blade, vein]
}
fn seed() -> Vec<Cell> {
    (0..N * N)
        .map(|i| {
            let x = 2. * (i % N) as f32 / (N - 1) as f32 - 1.;
            let y = 2. * (i / N) as f32 / (N - 1) as f32 - 1.;
            Cell {
                growth: if x.abs() < 0.09 && y > 0.65 { 0.2 } else { 0. },
                vein: 0.,
                x,
                y,
            }
        })
        .collect()
}
fn nbr(c: &[Cell], i: usize, ch: usize) -> f32 {
    let x = i % N;
    let y = i / N;
    let mut s = 0.;
    let mut n = 0.;
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            if let (Some(xx), Some(yy)) = (x.checked_add_signed(dx), y.checked_add_signed(dy)) {
                if xx < N && yy < N {
                    let v = c[yy * N + xx];
                    s += if ch == 0 { v.growth } else { v.vein };
                    n += 1.;
                }
            }
        }
    }
    if n > 0. { s / n } else { 0. }
}
fn features(c: &[Cell], i: usize, ch: usize) -> [f32; K] {
    let v = c[i];
    let g = v.growth;
    let vein = v.vein;
    [
        1.,
        v.x,
        v.y,
        v.x * v.x,
        v.y * v.y,
        g,
        vein,
        nbr(c, i, ch),
        v.x * v.y,
    ]
}
fn mask(i: usize, t: usize, seed: u64) -> bool {
    let mut z = (i as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (t as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ seed;
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58476d1ce4e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z % 100) < 75
}
fn activate(w: &[f32; K], f: &[f32; K]) -> f32 {
    w.iter().zip(f).map(|(a, b)| a * b).sum::<f32>().tanh()
}
fn step(old: &[Cell], new: &mut [Cell], w: &W, t: usize, runseed: u64, parallel: bool) {
    let calc = |i: usize| {
        let oldv = old[i];
        let mut v = oldv;
        if mask(i, t, runseed) {
            let dg = activate(&w[0], &features(old, i, 0));
            let dv = activate(&w[1], &features(old, i, 1));
            v.growth = (v.growth + DT * dg).clamp(0., 1.);
            v.vein = (v.vein + DT * dv).clamp(0., 1.);
        }
        v
    };
    if parallel {
        new.par_iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = calc(i));
    } else {
        for (i, v) in new.iter_mut().enumerate() {
            *v = calc(i);
        }
    }
}
fn rollout(w: &W, initial: &[Cell], runseed: u64, parallel: bool) -> Vec<Vec<Cell>> {
    let mut layers = vec![initial.to_vec()];
    for t in 0..STEPS {
        let mut next = vec![layers[t][0]; N * N];
        step(&layers[t], &mut next, w, t, runseed, parallel);
        layers.push(next);
    }
    layers
}
fn loss(v: &[Cell]) -> f32 {
    v.iter()
        .map(|c| {
            let z = target(c.x, c.y);
            (c.growth - z[0]).powi(2) + (c.vein - z[1]).powi(2) * 0.4
        })
        .sum::<f32>()
        / (N * N) as f32
}
fn objective(w: &W, damaged: bool) -> f32 {
    let mut initial = seed();
    if damaged {
        for (i, c) in initial.iter_mut().enumerate() {
            if (i % N) > N / 3 && (i % N) < 2 * N / 3 && i / N > N / 3 && i / N < 2 * N / 3 {
                c.growth = 0.;
                c.vein = 0.;
            }
        }
    }
    let layers = rollout(w, &initial, 17, false);
    (loss(layers.last().unwrap()) + 0.25 * loss(&layers[STEPS / 2])) / 1.25
}
fn train() -> W {
    let mut w = [[0f32; K]; 2];
    w[0][0] = 0.2;
    w[1][0] = 0.05;
    let now = Instant::now();
    let eps = 0.005;
    let mut velocity = [[0.; K]; 2];
    for epoch in 0..48 {
        let mut grad = [[0.; K]; 2];
        for ch in 0..2 {
            for k in 0..K {
                w[ch][k] += eps;
                let positive = objective(&w, epoch % 3 == 0);
                w[ch][k] -= 2. * eps;
                let negative = objective(&w, epoch % 3 == 0);
                w[ch][k] += eps;
                grad[ch][k] = (positive - negative) / (2. * eps);
            }
        }
        for ch in 0..2 {
            for k in 0..K {
                velocity[ch][k] = 0.8 * velocity[ch][k] + grad[ch][k];
                w[ch][k] -= RATE * velocity[ch][k];
            }
        }
        if epoch % 12 == 11 {
            println!("epoch {} objective {:.6}", epoch + 1, objective(&w, false));
        }
    }
    println!("training duration {:?}", now.elapsed());
    w
}
fn bench(w: &W) {
    let initial = seed();
    let runs = 7;
    for &threads in &[1usize, 2, 4, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for &parallel in &[false, true] {
            if !parallel && threads != 1 {
                continue;
            }
            let mut samples = Vec::new();
            let mut final_state = Vec::new();
            for r in 0..runs {
                let start = Instant::now();
                let layers = pool.install(|| rollout(w, &initial, 17, parallel));
                let duration = start.elapsed().as_secs_f64() * 1000.;
                if r == 0 {
                    final_state = layers.last().unwrap().clone();
                } else {
                    assert_eq!(final_state, layers.last().unwrap().clone());
                }
                samples.push(duration);
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "variant={} threads={} median_ms={:.4} min_ms={:.4} max_ms={:.4} loss={:.6}",
                if parallel { "rayon" } else { "sequential" },
                threads,
                samples[3],
                samples[0],
                samples[6],
                loss(&final_state)
            );
        }
    }
}
fn main() {
    let w = train();
    let initial = seed();
    let grown = rollout(&w, &initial, 17, false);
    let before = loss(&initial);
    let after = loss(grown.last().unwrap());
    let mut damaged = grown.last().unwrap().clone();
    for c in &mut damaged {
        if c.x.abs() < 0.28 && c.y.abs() < 0.25 {
            c.growth = 0.;
            c.vein = 0.;
        }
    }
    let damaged_loss = loss(&damaged);
    let healed = rollout(&w, &damaged, 17, false);
    println!(
        "target_fit initial_loss={:.6} final_loss={:.6} damaged_loss={:.6} regrown_loss={:.6}",
        before,
        after,
        damaged_loss,
        loss(healed.last().unwrap())
    );
    bench(&w);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parallel_exact() {
        let w = [[0.1; K]; 2];
        let init = seed();
        assert_eq!(
            rollout(&w, &init, 17, false).last(),
            rollout(&w, &init, 17, true).last()
        );
    }
    #[test]
    fn masks_deterministic() {
        assert_eq!(mask(41, 6, 17), mask(41, 6, 17));
    }
}