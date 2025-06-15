gm = GridPriceManager(1000, 800, 10)

# 底仓事件，在调用base_position方法时会触发此事件
@gm.event('base_position')
def base_position(price):
    # 传入最近的网格价格，以此价格作为买入价格参考
    print(price)
    return 123456	# 返回底仓订单，manger将订单记录