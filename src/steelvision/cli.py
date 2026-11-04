"""Interface de linha de comando para o projeto de inspeção de aço."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validar RLE e dividir imagens por hash")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    eda = commands.add_parser("eda", help="Exploração das classes e máscaras")
    eda.add_argument("--data", required=True)
    eda.add_argument("--manifest", required=True)
    eda.add_argument("--out", required=True)
    train = commands.add_parser("train", help="Treinar U-Net e selecionar na validação")
    train.add_argument("--data", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--height", type=int, default=128)
    train.add_argument("--width", type=int, default=800)
    train.add_argument("--base-channels", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    train.add_argument("--num-workers", type=int, default=0)
    evaluate = commands.add_parser("evaluate", help="Avaliar no teste reservado")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--device", default="auto")
    predict = commands.add_parser("predict", help="Inferência em uma imagem")
    predict.add_argument("--image", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--out", required=True)
    predict.add_argument("--device", default="auto")
    demo = commands.add_parser("demo", help="Executar fluxo completo com dados SINTÉTICOS")
    demo.add_argument("--out", default="artifacts/synthetic_demo")
    demo.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_manifest

        print(prepare_manifest(args.data, args.out, args.seed))
    elif args.command == "eda":
        from .visualize import explore

        print(json.dumps(explore(args.data, args.manifest, args.out), indent=2))
    elif args.command == "train":
        from .engine import Config, train

        config = Config(
            epochs=args.epochs,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            base_channels=args.base_channels,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            data_kind="synthetic" if (Path(args.data) / "SYNTHETIC.txt").exists() else "severstal",
        )
        print(train(args.data, args.manifest, args.out, config))
    elif args.command == "evaluate":
        from .engine import evaluate

        print(
            json.dumps(
                evaluate(args.data, args.manifest, args.checkpoint, args.out, device=args.device),
                indent=2,
            )
        )
    elif args.command == "predict":
        from .engine import predict

        result = predict(args.image, args.checkpoint, args.out, args.device)
        print(f"Predição salva: {result['image_id']}")
    elif args.command == "demo":
        import torch

        from .data import prepare_manifest
        from .engine import Config, evaluate, predict, train
        from .synthetic import generate_synthetic
        from .visualize import explore

        # Smoke test pequeno e previsível mesmo sem acelerador.
        torch.set_num_threads(2)
        root = Path(args.out)
        data = generate_synthetic(root / "data")
        manifest = prepare_manifest(data, root / "prepared")
        explore(data, manifest, root / "eda", size=(64, 256))
        config = Config(
            epochs=args.epochs,
            height=64,
            width=256,
            base_channels=8,
            device="cpu",
            data_kind="synthetic",
            batch_size=4,
        )
        checkpoint = train(data, manifest, root / "run", config)
        evaluate(data, manifest, checkpoint, root / "evaluation", device="cpu")
        predict(
            next((data / "train_images").glob("*.png")),
            checkpoint,
            root / "inference",
            device="cpu",
        )
        print(f"DEMO SINTÉTICA concluída: {root.resolve()}")


if __name__ == "__main__":
    main()
