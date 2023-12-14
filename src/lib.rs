use std::{collections::HashMap, fs::File, io::Read};
use polars::prelude::*;
use rayon::prelude::*;
use serde_json::*;


#[cfg(test)]
mod tests;


fn read_vocab_dt() -> DataFrame {
    // reading in the vocab dt
    let result = CsvReader::from_path("./proc/bpe_vocab.tsv")
        .expect("File not found!")
        .with_separator("\t".as_bytes()[0])
        .has_header(true)
        .finish()
        .expect("PolarsError");

    return result;
}


// extract columns and returns as vectors.
fn extract_columns(vocab_dt: DataFrame) -> (Vec<i64>, Vec<String>) {

    let token: Vec<i64> = vocab_dt
        .column("token")
        .unwrap()
        .i64()
        .unwrap()
        .into_iter()
        .flatten()
        .collect();

    let sequence: Vec<String> = vocab_dt
        .column("sequence")
        .expect("Problems in getting the column")
        .utf8()
        .unwrap()
        .into_iter()
        .flatten()
        .map(|seq| seq.to_owned())
        .collect();


    return (token, sequence);
}


fn build_vocab_hashmap(token: Vec<i64>, sequence: Vec<String>) -> HashMap<i64, String> {

    let vocab_hashmap: HashMap<i64, String> = token.iter()
        .zip(sequence.iter())
        .into_iter()
        .map(|x| (x.0.clone(), x.1.clone()))
        .collect();
    

    return vocab_hashmap;
}


fn serialize(vocab_hashmap: HashMap<i64, String>) {
    
    let target = File::create("./proc/vocab_hashmap.json")
        .expect("Cannot open file for writing!");
    serde_json::to_writer(target, &vocab_hashmap)
        .expect("Cannot serialize and save it as json");
}


fn deserialize(filepath: String) -> HashMap<i64, String> {
   
    let mut target = File::open(filepath)
        .expect("Cannot open file. Check file path");
    let mut json_text = String::new();

    target.read_to_string(&mut json_text)
        .expect("Cannot read in json file");


    serde_json::from_str(&json_text)
        .expect("Cannot deserialize json text")
}


fn get_sequence_from_token(vocab_hashmap: HashMap<i64, String>,
                           token: i64) -> String {

    match vocab_hashmap.get(&token) {
       Some(seq) => return seq.to_owned(),
       None => return String::from("")
    }
}


fn decode(tensor: Vec<i64>,
          vocab_hashmap: HashMap<i64, String>) -> String {

    let decoded_seq: String = tensor
        .par_iter()
        .map(|curr_token| {
            match vocab_hashmap.get(&curr_token) {
                Some(seq) => return seq.to_owned(),
                None => return String::from("")
            }
        })
        .collect::<Vec<String>>()
        .join("");

    return decoded_seq;
}


fn decode_batch(batch: Vec<Vec<i64>>,
                vocab_hashmap: HashMap<i64, String>) -> Vec<String>{
   
    let batch_seq: Vec<String> = batch
        .par_iter()
        .map(|curr_tensor| decode(curr_tensor.to_owned(), vocab_hashmap.to_owned()))
        .collect();

    return batch_seq;
}
