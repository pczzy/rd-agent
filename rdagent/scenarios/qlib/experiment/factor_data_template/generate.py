import qlib

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

from qlib.data import D

instruments = D.instruments()
fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":].sort_index()

data.to_hdf("./daily_pv_all.h5", key="data")


fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
debug_data = (
    D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
    .swaplevel()
    .sort_index()
)

# Instruments delisted before the debug window exist in the full history but have no rows
# here, so intersect first, otherwise `.loc` raises KeyError on the missing ones.
available = set(debug_data.index.get_level_values("instrument"))
selected = [i for i in data.index.get_level_values("instrument").unique() if i in available][:100]

debug_data = debug_data.swaplevel().sort_index().loc[selected].swaplevel().sort_index()

debug_data.to_hdf("./daily_pv_debug.h5", key="data")
