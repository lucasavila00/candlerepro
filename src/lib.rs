use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::Device;
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    pub use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    pub n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    pub normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation.
    #[arg(long, default_value = "false")]
    pub approximate_gelu: bool,
}

impl Args {
    pub fn build_model_and_tokenizer(&self) -> Result<AiCtx> {
        let cpu = false;
        let device = device(cpu)?;

        let model_id = "BAAI/bge-base-en-v1.5".to_string();

        let repo = Repo::with_revision(model_id, RepoType::Model, "main".to_string());
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        if self.approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        let model = BertModel::load(vb, &config)?;
        Ok(AiCtx { model, tokenizer })
    }
}
pub fn build_ai_ctx() -> Result<AiCtx> {
    let args = Args {
        prompt: None,
        use_pth: false,
        n: 1,
        normalize_embeddings: true,
        approximate_gelu: false,
    };
    let ai_ctx = args.build_model_and_tokenizer()?;
    Ok(ai_ctx)
}
pub struct AiCtx {
    model: BertModel,
    tokenizer: Tokenizer,
}
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

impl AiCtx {
    
    pub fn txt_to_vec_many(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let tensor = self.tensor_of_text_many(texts)?;

        let embeddings = tensor.get_on_dim(1, 0)?;
        let embeddings = normalize_l2(&embeddings)?;

        let mut accum = vec![];
        for i in 0..texts.len() {
            let vec1: Vec<f32> = embeddings.get(i).unwrap().to_vec1()?;
            accum.push(vec1);
        }
        Ok(accum)
    }
    pub fn txt_to_vec(&mut self, txt: &str) -> Result<Vec<f32>> {
        let tensor = self.tensor_of_text(&txt)?;
        let embeddings = tensor.get_on_dim(1, 0)?;
        let embeddings = normalize_l2(&embeddings)?;
        let vec1: Vec<f32> = embeddings.get(0).unwrap().to_vec1()?;
        Ok(vec1)
    }
    fn tensor_of_text(&mut self, prompt: &str) -> Result<Tensor> {
        // let start = std::time::Instant::now();
        let tokenizer = self
            .tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        // println!("Tokenizer took {:?}", start.elapsed());

        // let start = std::time::Instant::now();
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // tokens.push(0);

        // println!("Tokens took {:?}", start.elapsed());
        let token_ids = Tensor::new(&tokens[..], &self.model.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        // println!("Loaded and encoded {:?}", start.elapsed());
        // let start = std::time::Instant::now();
        let ys = self.model.forward(&token_ids, &token_type_ids)?;
        // println!("FWD took {:?}", start.elapsed());
        Ok(ys)
    }

    fn tensor_of_text_many(&mut self, sentences: &[&str]) -> Result<Tensor> {
        // let n_sentences = sentences.len();

        let tokenizer = self
            .tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        let tokens = tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.model.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        println!("tokens:");
        let token_ids = Tensor::stack(&token_ids, 0)?;
        println!("{}", token_ids);
        println!("");
        let token_type_ids = token_ids.zeros_like()?;

        let ys = self.model.forward(&token_ids, &token_type_ids)?;
        // println!("FWD took {:?}", start.elapsed());
        Ok(ys)
    }
}
