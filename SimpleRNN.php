<?php
set_time_limit(30000);
function Error($output, $true_label) {
    return ($output - $true_label);
}

function Errors($outputs, $true_labels) {
    if(count($outputs) != count($true_labels)) throw New Exception("Error MissMatched Arguments");
    $response = [];
    foreach($outputs as $key => $output) {
        $response[] = ($output - $true_labels[$key]);
    }
    return $response;
}

function MatrixMultiplication($inputs, $weights) {
    if(count($inputs) != count($weights)) throw New Exception("Error MissMatched Arguments");
    $response = 0;
    foreach($weights as $key => $weight) {
        $response+= $inputs[$key] * $weight;
    }
    return $response;
}

function UpdateWeight($error, $weights, $inputs, $prediction) {
    global $inputDimensions;
    if(count($inputs) != count($weights)) throw New Exception("Error MissMatched Arguments");
    $response = $weights;
    foreach($weights as $key => $weight) {
        $rate_of_change = ReluDrivative($prediction) *  $inputs[$key] * $error * 0.00001;
        $response[$key]+= $rate_of_change;
        if(is_nan($response[$key]) || is_infinite($response[$key])) {
            $response[$key] = (float) rand() 
            / (float) getrandmax(); 
        }
    }
    return $response;
}


function Train($iterations = 10000) {
    global $bias, $inputs, $weights, $outputs;
    for ($i = 0; $i < $iterations; $i++) {
    
        foreach ($inputs as $key => $input) {
            $errors = [];
            $perdiction = [];
            $temp = $input;
            foreach ($outputs[$key] as $index => $output) {
                $appened_predictions = MatrixMultiplication($temp, $weights[$index]) + $bias[$index];
                $perdiction[$index] = $appened_predictions;
                
                
                $perdiction[$index] = Relu($perdiction[$index]);
            
                $errors[$index] = Error($output, $perdiction[$index]);
                $weights[$index] = UpdateWeight($errors[$index], $weights[$index], $temp, $perdiction[$index]);
                $temp = $input;
                array_push($temp, $perdiction[$index]);
            }
          
        }
    }
}

function Relu($x) {
    return $x <= 0? 0: $x;
}

function ReluDrivative($x) {
    return $x < 0? 0: 1;
}


function FeedForward($input) {
    global $bias, $inputs, $weights;
    $perdiction = [];
    $temp = $input;
    
    foreach ($weights as $key => $weight) {
        $appened_predictions = MatrixMultiplication($temp, $weight) + $bias[$key];
        $perdiction[$key] = $appened_predictions;
        $perdiction[$key] = Relu($perdiction[$key]);
        $temp = $input;
        array_push($temp, $perdiction[$key]);
        
    }
    return $perdiction;

}

function customDivision($inputArray) {
    $count = count($inputArray);

    foreach ($inputArray as &$value) {
        // If count is zero, set it to 1 to avoid division by zero
        $count = max($count, 1);

        $value /= $count;
        $count--;
    }

    return $inputArray;
}

// Create word-to-index mapping
function GenerateWordIndex($Dataset) {
    $response = [];
    $index = 1; // Start index from 1
    foreach ($Dataset as $data) {
        $userQueryWords = str_split($data[0], 1);
        $chatbotResponseWords = str_split($data[1], 1);
    
        foreach (array_merge($userQueryWords, $chatbotResponseWords) as $word) {
            $word = strtolower($word);
            if (!isset($wordIndexMapping[$word])) {
                $response[$word] = $index;
                $index++;
            }
        }
    }
    return $response;
}

function WordToIndex($data) {
    global $wordIndexMapping;
    return array_map(function ($word) use ($wordIndexMapping) {
        $word = strtolower($word);
        return $wordIndexMapping[$word]/count($wordIndexMapping);
    }, str_split($data, 1));
    
}

function TruncatAndPadding($array, $limit) {
    return array_pad(array_slice($array, 0, $limit), $limit, 0);
}

function curateDataset() {
    global $inputs, $outputs, $inputDimensions, $outputDimensions, $Dataset;
    foreach ($Dataset as $data) {
        // Tokenize user query
        $userQueryTokens = WordToIndex($data[0]);
    
        // Tokenize chatbot response
        $chatbotResponseTokens = WordToIndex($data[1]);
    
        // Pad or truncate to the fixed length
        $inputs[] = TruncatAndPadding($userQueryTokens, $inputDimensions);
        
        $outputs[] = TruncatAndPadding($chatbotResponseTokens, $outputDimensions);
    }
}

function GenerateBias() {
    global $bias, $outputDimensions;
    // Generate random biases
    $response = [];
    for ($i = 0; $i < $outputDimensions; $i++) {
        $response[] = rand(1, 10); // Adjust the range as needed
    }
    return $response;
}

function normalizeArray($arr) {
    $min_val = min($arr);
    $max_val = max($arr);

    $normalized_arr = array_map(function ($x) use ($min_val, $max_val) {
        return ($x - $min_val) / ($max_val - $min_val);
    }, $arr);

    return $normalized_arr;
}

function findClosestMatch($searchValue, $array) {
    $closest = null;
    $minDifference = PHP_INT_MAX;

    foreach ($array as $value) {
        $difference = abs($searchValue - $value);

        if ($difference < $minDifference) {
            $minDifference = $difference;
            $closest = $value;
        }
    }

    return $closest;
}

// Generate random weights
function GenerateWeights() {
    global $weights, $inputDimensions, $outputDimensions;
    $response = [];
    $recursive_limit = $inputDimensions;
    for ($i = 0; $i < $outputDimensions; $i++) {
        $response[$i] = [];
        if($i == 1) {
            $recursive_limit++;
        }
        for ($j = 0; $j < $recursive_limit; $j++) {
            $response[$i][] = (float) rand() 
            / (float) getrandmax(); 
            // Random float between 0 and 1
        }
    }
    return $response;
}

function saveModel($weights, $bias, $filePath = 'model.json') {
    $modelData = [
        'weights' => $weights,
        'bias' => $bias,
    ];

    $jsonData = json_encode($modelData, JSON_PRETTY_PRINT);

    if (file_put_contents($filePath, $jsonData) === false) {
        throw new Exception("Error saving the model to $filePath");
    }

    echo "Model saved successfully to $filePath\n";
}

function CalculateAccuracy($predictions, $trueLabels) {
    $correctPredictions = 0;

    foreach ($predictions as $key => $prediction) {
        // Assuming $prediction and $trueLabels are arrays of equal length
        if ($prediction === $trueLabels[$key]) {
            $correctPredictions++;
        }
    }

    $totalPredictions = count($predictions);
    $accuracy = $correctPredictions / $totalPredictions * 100;

    return $accuracy;
}

function loadModel($filePath = 'model.json') {
    if (!file_exists($filePath)) {
        throw new Exception("Model file $filePath not found");
    }

    $jsonData = file_get_contents($filePath);
    if ($jsonData === false) {
        throw new Exception("Error reading the model from $filePath");
    }

    $modelData = json_decode($jsonData, true);

    if ($modelData === null) {
        throw new Exception("Error decoding JSON from $filePath");
    }

    return $modelData;
}


global $bias, $inputs, $weights, $outputs, $inputDimensions, $outputDimensions, $wordIndexMapping, $Dataset;
$Dataset = [
    ['Hi there.', 'How can I help you?'],
        ['What time is it', 'The time is CURRENTTIME.'],
    ['How is the weather today?', 'Chatbot: The weather is currently 15 degrees Celsius with a slight breeze.'],
    ['Tell me a joke.', 'Chatbot: Why don’t scientists trust atoms? Because they make up everything!'],
    ['What is the capital of France?', 'Chatbot: The capital of France is Paris.'],
      ['What\'s the latest news?', 'Chatbot: In recent updates, scientists have made progress in renewable energy research, paving the way for a sustainable future.'],

['Tell me a fun fact!', 'Chatbot: Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.'],

['Any movie recommendations?', 'Chatbot: I suggest watching "Inception." Its mind-bending plot and stunning visuals make it a must-watch for any movie enthusiast.'],

['How can I improve my productivity?', 'Chatbot: Consider trying the Pomodoro Technique. Work for 25 minutes, then take a 5-minute break. Repeat this cycle to enhance focus and efficiency.'],

['What\'s a good recipe for dinner?', 'Chatbot: How about trying a simple and delicious chicken stir-fry? Sauté chicken, veggies, and your favorite sauce for a quick and tasty meal.'],

['Share a book recommendation, please.', 'Chatbot: "The Alchemist" by Paulo Coelho is a timeless classic that explores the journey of self-discovery and pursuing one\'s dreams.'],

['Tell me a joke!', 'Chatbot: Why don\'t scientists trust atoms? Because they make up everything!'],

['Any travel tips for a weekend getaway?', 'Chatbot: Consider exploring a nearby town or city you haven\'t visited before. It\'s a great way to have a mini-adventure without extensive planning.'],

['What\'s the best way to stay motivated?', 'Chatbot: Set specific, achievable goals, and celebrate small victories along the way. This can help maintain motivation and a sense of accomplishment.'],

['Can you recommend a podcast?', 'Chatbot: "The Daily" by The New York Times offers insightful daily news coverage, providing a convenient way to stay informed on current events.'],

];


$inputDimensions = 20; // Example input dimensions
$outputDimensions = 100; // Example output 

$wordIndexMapping = GenerateWordIndex($Dataset);

// Tokenize and pad/truncate sentences
$inputs = [];
$outputs = [];

$new_model = false;
curateDataset();

if($new_model) {
    $weights = GenerateWeights();
    $bias = GenerateBias();
}
else {
    // Load the model
    $loadedModel = loadModel('saved_model.json');
    $weights = $loadedModel['weights'];
    $bias = $loadedModel['bias'];
}

// Now you can use $loadedWeights and $loadedBias for predictions or further training.

Train();

$UserInput = $inputs[2];
$perdictions = FeedForward($UserInput);

$trueLabels = $outputs[2];

$accuracy = CalculateAccuracy($perdictions, $trueLabels);

echo "Accuracy: $accuracy%";

//print_r($weights);

echo "<pre>";

$indexWordMapping = array_flip($wordIndexMapping);
$words = [];
foreach ($perdictions as $key => $index) {
    $value = $index * count($wordIndexMapping);
    $words[] = $indexWordMapping[$value];
}

// Display the converted words
print_r($words);

// Example usage:
// Save the model
saveModel($weights, $bias, 'saved_model.json');
$url1=$_SERVER['REQUEST_URI'];
header("Refresh: 5; URL=$url1");
echo "</pre>";
