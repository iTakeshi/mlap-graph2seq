import argparse

from mlap.main import main


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--no-save"     , action="store_true" , help="do not save trained model parameters"                        )
    parser.add_argument( "--train"       , action="store_true" , help="run training"                                                )
    parser.add_argument( "--test"        , type=str            , help="run test on specified models"         , nargs="+"            )
    parser.add_argument( "--dataset"     , type=str            , help="dataset name"                         , required=True        )
    parser.add_argument( "--arch"        , type=str            , help="network architecture"                 , default="gin-simple" )
    parser.add_argument( "--norm"        , type=str            , help="normalization method"                 , default="none"       )
    parser.add_argument( "--residual"    , action="store_true" , help="whether to use residual connection"                          )
    parser.add_argument( "--dim-feat"    , type=int            , help="number of dimensions of GNN"          , default=200          )
    parser.add_argument( "--depth"       , type=int            , help="number of layers of GNN"              , default=5            )
    parser.add_argument( "--epochs"      , type=int            , help="number of training epochs"            , default=50           )
    parser.add_argument( "--batch-size"  , type=int            , help="batch size"                           , default=50           )
    parser.add_argument( "--initial-lr"  , type=float          , help="initial learning late"                , default=1e-3         )
    parser.add_argument( "--lr-interval" , type=int            , help="number of epochs to decay LR"         , default=15           )
    parser.add_argument( "--lr-scale"    , type=float          , help="scale for LR decay"                   , default=0.2          )
    parser.add_argument( "--seed"        , type=int            , help="seed"                                 , default=0            )

    parser.add_argument( "--code2-use-subtoken" , action="store_true" , help="subtokenize node attrs in ogbg-code2 preprocess"                    )
    parser.add_argument( "--code2-decoder-type" , type=str            , help="decoder type for ogbg-code2 (linear or lstm)"    , default="linear" )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(
        str(args),
        args.dataset,
        args.batch_size,
        args.arch,
        args.norm,
        args.residual,
        args.dim_feat,
        args.depth,
        args.seed,
        args.epochs,
        args.initial_lr,
        args.lr_interval,
        args.lr_scale,
        train=args.train,
        test=args.test,
        save=not args.no_save,
        code2_use_subtoken=args.code2_use_subtoken,
        code2_decoder_type=args.code2_decoder_type,
    )
