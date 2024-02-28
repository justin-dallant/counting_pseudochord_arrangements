mod arr_counter;

use arr_counter::count;
use clap::Parser;
use colored::*;
use regex::Regex;

use std::env;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(
    help_template = " {author-with-newline} {about-section}Version: {version} \n\n {usage-heading} {usage} \n {all-args} {tab}"
)]
struct Args {
    /// Chords as pairs of integers between quotation marks. E.g. "(0,3), (1,4), (2,5)".
    #[arg(short, long)]
    chords: String,

    /// (Optional) Max time, in seconds, before atempting to abort the computation.
    #[arg(short, long)]
    timeout: Option<i64>,

    /// (Optional) Number of threads to spawn (default: 1).
    #[arg(short, long)]
    num_threads: Option<usize>,
}

fn parse_chords_string(s: &str) -> Result<Vec<(usize,usize)>, &str>{
    let s_no_whitespace: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let re = Regex::new(r"^[\(\[]?(\((0|[1-9][0-9]*),(0|[1-9][0-9]*)\),)*\((0|[1-9][0-9]*),(0|[1-9][0-9]*)\)[\)\]]?$").unwrap();
    if !re.is_match(&s_no_whitespace){
        Err("Can't parse chords.")
    }else{
        Ok(s_no_whitespace
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap()).collect::<Vec<usize>>()
        .chunks(2).map(|s| (s[0],s[1])).collect())
    }
}

fn main() {

    let args: Vec<String> = env::args().collect();
    let clap_args = Args::parse();

    let chords = parse_chords_string(&clap_args.chords);

    let num_threads: usize;
    match clap_args.num_threads{
        Some(t) => num_threads = t,
        None => num_threads = 1,
    }

    match chords{
        Err(err) => eprintln!("{} {} Example usage: {} --chords '(0,3), (1,4), (2,5)'", "error".red().bold(), err, args[0]),
        Ok(chords) => {
            rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global().unwrap();
            let res = count(chords, clap_args.timeout);
            match res {
                Some(c) => println!("Result: {}", c),
                None => {println!("{}", "Aborted (timeout).".red().bold()); println!("Result: 0")},
            }
            
        }
    };
}
