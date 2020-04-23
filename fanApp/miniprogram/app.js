//app.js
App({
  onLaunch: function () {
    
    if (!wx.cloud) {
      console.error('请使用 2.2.3 或以上的基础库以使用云能力')
    } else {
      wx.cloud.init({
        // env 参数说明：
        //   env 参数决定接下来小程序发起的云开发调用（wx.cloud.xxx）会默认请求到哪个云环境的资源
        //   此处请填入环境 ID, 环境 ID 可打开云控制台查看
        //   如不填则使用默认环境（第一个创建的环境）
        // env: 'my-env-id',
        traceUser: true,
      })
    }

    this.globalData = {
      input_fileName:"data_602541757299269579.csv",
      access_token: "a26cf5ac6aa64c9da06b43cc46271fac.7a4b961c2e7dd6d1386462a71948d3d1",
      output_fileName:'',
      resultArray: []
    }
  }
})
