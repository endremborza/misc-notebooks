import datetime as dt
import itertools

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd

from gluonts.model.deepar import DeepAREstimator
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import Predictor
from gluonts.trainer import Trainer
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator


from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AdhocTransform,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetFieldIfNotPresent,
    TransformedDataset,
)


class MockGluonData:
    def __init__(
        self,
        end_time="2017-06-01",
        get_start_time: callable = lambda: np.random.choice(
            pd.date_range("2017-01-01", "2017-03-01")
        ),
        dynamic_reals: int = 3,
        static_reals: int = 2,
        dynamic_cats: int = 2,
        static_cats: int = 2,
        freq: str = "D",
        base_shift: int = 7,
        dyn_cat_count: int = 3,
        stat_cat_count_range=(2, 6),
    ):

        self.date_col = "DATE"
        self.target_col = "target"
        self.freq = freq
        self.base_shift = base_shift
        self.dynamic_real_cols = [f"DR_{i}" for i in range(dynamic_reals)]
        self.static_real_cols = [f"SR_{i}" for i in range(static_reals)]
        self.dynamic_cat_cols = [f"DC_{i}" for i in range(dynamic_cats)]
        self.static_cat_cols = [f"SC_{i}" for i in range(static_cats)]

        self.full_df = self._get_full_df(
            end_time, get_start_time, dyn_cat_count, stat_cat_count_range
        )

    def _get_full_df(
        self,
        end_time,
        get_start_time: callable,
        dyn_cat_count: int,
        stat_cat_size_range: tuple,
    ):
        return pd.concat(
            [
                (
                    pd.DataFrame(
                        {
                            self.date_col: pd.date_range(
                                get_start_time(), end=end_time, freq=self.freq
                            )
                        }
                    )
                    .pipe(
                        lambda df: pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    np.random.rand(
                                        df.shape[0], len(self.dynamic_real_cols)
                                    ),
                                    columns=self.dynamic_real_cols,
                                ),
                                pd.DataFrame(
                                    {
                                        c: np.random.rand()
                                        for c in self.static_real_cols
                                    },
                                    index=df.index,
                                ),
                                pd.DataFrame(
                                    np.random.randint(
                                        1,
                                        dyn_cat_count + 1,
                                        size=(df.shape[0], len(self.dynamic_cat_cols)),
                                    ),
                                    columns=self.dynamic_cat_cols,
                                ),
                            ],
                            axis=1,
                        )
                    )
                    .assign(
                        **{
                            self.target_col: lambda df: df.drop(self.date_col, axis=1)
                            .shift(self.base_shift)
                            .pipe(lambda _df: df.sum(axis=1) / (_df.prod(axis=1) + 0.5))
                            .pipe(
                                lambda s: s
                                + s * np.random.rand() * 2
                                + s.shift(1) * np.random.rand()
                            )
                            .pipe(lambda s: s / s.mean())
                        }
                    )
                )
                .assign(
                    **{
                        scc: scv
                        for scc, scv in zip(self.static_cat_cols, stat_cat_values)
                    }
                )
                .dropna()
                for stat_cat_values in itertools.product(
                    *[
                        range(np.random.randint(*stat_cat_size_range))
                        for _ in self.static_cat_cols
                    ]
                )
            ]
        )

    def get_gluon_wrapped(self):

        return DataFrameGluonWrapper(
            df=self.full_df,
            freq=self.freq,
            target_col=self.target_col,
            date_col=self.date_col,
            static_cat_cols=self.static_cat_cols,
            static_real_cols=self.static_real_cols,
            dynamic_cat_cols=self.dynamic_cat_cols,
            dynamic_real_cols=self.dynamic_real_cols
        )


@dataclass
class DataFrameGluonWrapper:

    df: pd.DataFrame
    freq: str
    target_col: str
    date_col: str
    static_cat_cols: list
    dynamic_real_cols: list = field(default_factory=list)
    dynamic_cat_cols: list = field(default_factory=list)
    static_real_cols: list = field(default_factory=list)

    def get_list_dataset(self, df_filter: Union[callable, slice] = slice(None)):
        ds = ListDataset(
            [
                {
                    FieldName.TARGET: gdf.loc[:, self.target_col].values,
                    FieldName.START: gdf.loc[:, self.date_col].min(),
                    FieldName.FEAT_DYNAMIC_REAL: list(
                        gdf.loc[:, self.dynamic_real_cols].values.T
                    ),
                    FieldName.FEAT_DYNAMIC_CAT: list(
                        gdf.loc[:, self.dynamic_cat_cols].values.T
                    ),
                    FieldName.FEAT_STATIC_CAT: gid,
                    FieldName.FEAT_STATIC_REAL: gdf.loc[:, self.static_real_cols]
                    .iloc[0, :]
                    .values,
                }
                for gid, gdf in self.df.loc[df_filter, :].groupby(self.static_cat_cols)
            ],
            freq=self.freq,
        )

        static_cat_nums = (
            self.df.loc[df_filter, self.static_cat_cols].nunique().tolist()
        )
        return ds, static_cat_nums

    def get_rolling_preds(
        self,
        period_start: str,
        period_end: str,
        predictor: Predictor,
        context_length: int = 0,
        fixed_start: bool = True,
        wrapper=lambda x: x,
    ):
        pred_dfs = []
        context_length = context_length or predictor.prediction_net.context_length
        prediction_length = predictor.prediction_length

        for act_test_start_date in wrapper(
            pd.date_range(
                pd.Timestamp(period_start) + dt.timedelta(context_length),
                pd.Timestamp(period_end) - dt.timedelta(prediction_length),
            )
        ):
            act_context_start = (
                pd.Timestamp(period_start)
                if fixed_start
                else pd.Timestamp(act_test_start_date) - dt.timedelta(context_length)
            )
            act_test_ds, _ = self.get_list_dataset(
                lambda df: (
                    (df[self.date_col] >= act_context_start)
                    & (
                        df[self.date_col]
                        < pd.Timestamp(act_test_start_date)
                        + dt.timedelta(prediction_length)
                    )
                )
            )

            dataset_trunc = TransformedDataset(
                act_test_ds,
                transformations=[
                    AdhocTransform(
                        lambda data: {
                            **data,
                            FieldName.TARGET: data[FieldName.TARGET][
                                ..., :-prediction_length
                            ],
                        }
                    )
                ],
            )

            forecasts = predictor.predict(dataset_trunc, num_samples=500)

            for forecast_entry, test_entry in zip(forecasts, act_test_ds):
                pred_dfs.append(
                    pd.DataFrame(
                        {
                            "pred_y": forecast_entry.mean,
                            "test_y": test_entry[FieldName.TARGET][-prediction_length:],
                            "live": range(-1, -prediction_length - 1, -1),
                            self.date_col: pd.date_range(
                                start=test_entry[FieldName.START],
                                freq=self.freq,
                                periods=len(test_entry[FieldName.TARGET]),
                            )[-prediction_length:],
                        }
                    ).assign(
                        **{
                            k: v
                            for k, v in zip(
                                self.static_cat_cols,
                                test_entry[FieldName.FEAT_STATIC_CAT],
                            )
                        }
                    )
                )
        return pd.concat(pred_dfs)
