# 创建了一个价格1000-800，数量为10的等差网格
GridPriceManager(upper_price=1000, lower_price=800, grid_num=10)

# 创建了一个价格1000-800，间隔为1%的等比网格
GridPriceManager(upper_price=1000, lower_price=800, interval=1)

# 传入已有网格信息
data = {
	"grid_list":    {99:None,100:None,101:None,102:None,103:None,104:None},
	"interval":     None,
	"upper_price":  104,
	"lower_price":  99,
	"grid_num":     6,
	"side":         "long",
	"grid_diff":    1,
	"type":         "等差",
}
GridPriceManager(Data=data)