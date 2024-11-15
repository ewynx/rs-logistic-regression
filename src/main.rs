fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn predict(features: &[f64], weights: &[f64]) -> f64 {
    sigmoid(dot_product(features, weights))
}

fn compute_gradient(
    features: &[Vec<f64>], // Each row is a data point
    labels: &[f64],        // Corresponding labels
    weights: &[f64],       // Current weights
) -> Vec<f64> {
    let m = features.len() as f64;

    // Compute gradient for each weight
    let mut gradients = vec![0.0; weights.len()];
    for i in 0..features.len() {
        let h = sigmoid(dot_product(&features[i], weights));
        let error = h - labels[i];
        for j in 0..weights.len() {
            gradients[j] += error * features[i][j];
        }
    }

    // Average the gradients
    gradients.iter().map(|g| g / m).collect()
}
fn update_weights(weights: &mut [f64], gradients: &[f64], learning_rate: f64) {
    for (w, g) in weights.iter_mut().zip(gradients.iter()) {
        *w -= learning_rate * g;
    }
}
fn train(
    features: &[Vec<f64>],
    labels: &[f64],
    mut weights: Vec<f64>,
    learning_rate: f64,
    epochs: usize,
) -> Vec<f64> {
    for _ in 0..epochs {
        let gradients = compute_gradient(features, labels, &weights);
        update_weights(&mut weights, &gradients, learning_rate);
    }
    weights
}

fn main() {
    println!("Hello, world!");
}

#[test]
fn test_setosa_versicolor() {
    // Define the dataset (features and labels)
    let inputs = vec![
        vec![5.1, 3.5, 1.4, 0.2], // Setosa
        vec![4.9, 3.0, 1.4, 0.2], // Setosa
        vec![4.7, 3.2, 1.3, 0.2], // Setosa
        vec![4.6, 3.1, 1.5, 0.2], // Setosa
        vec![5.0, 3.6, 1.4, 0.2], // Setosa
        vec![7.0, 3.2, 4.7, 1.4], // Versicolor
        vec![6.4, 3.2, 4.5, 1.5], // Versicolor
        vec![6.9, 3.1, 4.9, 1.5], // Versicolor
        vec![5.5, 2.3, 4.0, 1.3], // Versicolor
        vec![6.5, 2.8, 4.6, 1.5], // Versicolor
    ];

    // Labels: Setosa (0), Versicolor (1)
    let labels = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    // Initialize weights
    let weights = vec![0.0; inputs[0].len()];
    let learning_rate = 0.1;
    let epochs = 1000;

    // Train the model
    let trained_weights = train(&inputs, &labels, weights, learning_rate, epochs);

    // Make predictions
    println!("Trained Weights: {:?}", trained_weights);
    for (i, input) in inputs.iter().enumerate() {
        let prediction = predict(input, &trained_weights);
        println!(
            "Data Point: {:?}, True Label: {}, Predicted: {}",
            input, labels[i], prediction
        );
    }
    /*
    Trained Weights: [-0.5970033489455933, -2.002002350313541, 2.9098182502323118, 1.3413750389637702]
    Data Point: [5.1, 3.5, 1.4, 0.2], True Label: 0, Predicted: 0.0033027067470813103
    Data Point: [4.9, 3.0, 1.4, 0.2], True Label: 0, Predicted: 0.010057752228614993
    Data Point: [4.7, 3.2, 1.3, 0.2], True Label: 0, Predicted: 0.0057016340013520414
    Data Point: [4.6, 3.1, 1.5, 0.2], True Label: 0, Predicted: 0.013132829233189409
    Data Point: [5.0, 3.6, 1.4, 0.2], True Label: 0, Predicted: 0.002871043230476319
    Data Point: [7.0, 3.2, 4.7, 1.4], True Label: 1, Predicted: 0.9930954067967673
    Data Point: [6.4, 3.2, 4.5, 1.5], True Label: 1, Predicted: 0.9924528966911029
    Data Point: [6.9, 3.1, 4.9, 1.5], True Label: 1, Predicted: 0.9973870044781479
    Data Point: [5.5, 2.3, 4.0, 1.3], True Label: 1, Predicted: 0.995909286319737
    Data Point: [6.5, 2.8, 4.6, 1.5], True Label: 1, Predicted: 0.9972981098823148
      */
}
