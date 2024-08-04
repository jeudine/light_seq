# LSeq

[Raspberry Pi](https://www.raspberrypi.com/) based light show program reacting to the audio input.

## Installation

### Cargo

If you already have a Rust environment set up, you can use the `cargo install` command:

```bash
cargo install lseq
```

## Usage

### Connection

Connect your lights to GPIO 0, 1 and 2 of the Raspberry Pi.

### Execution

You can either run the program without any command line parameters and select the audio device during the execution (`lseq`) or you can specify the `id` of the audio interface in the command line (e.g. to select the audio interface 0, run `lseq 0`) .

During the execution you can switch between 4 modes. To switch to a given mode, type the character corresponding to the mode and press enter.

| Mode     | Character |
| ---      | ---       |
| Lights reacting to audio input | `0` |
| Testing | `1` |
| Lights On | `2` |
| Lights Off | `3` |

The default mode when the program starts is `Lights On`.

To quit the program type `=` and then press enter.
