use anyhow::Result;
use candlereprobert::build_ai_ctx;

fn main() -> Result<()> {
    let txt = "The cat sits outside";
    let mut ai_ctx = build_ai_ctx()?;
    let r = ai_ctx.txt_to_vec(txt)?;
    // dbg fist 5, last 5

    println!("one at a time");
    println!("{:?}", &r[..5]);
    println!("{:?}", &r[r.len() - 5..]);

    println!("one at a time, but batched");
    let r = ai_ctx.txt_to_vec_many(&[txt])?;
    println!("{:?}", &r[0][..5]);
    println!("{:?}", &r[0][r[0].len() - 5..]);

    println!("two at a time, first");
    let r = ai_ctx.txt_to_vec_many(&[
        txt,
        //
        "some other text that will use padding",
    ])?;
    println!("{:?}", &r[0][..5]);
    println!("{:?}", &r[0][r[0].len() - 5..]);

    println!("two at a time, second");
    println!("{:?}", &r[1][..5]);
    println!("{:?}", &r[1][r[0].len() - 5..]);

    Ok(())
}
