function formatTime(date) {
  var year = date.getFullYear()
  var month = date.getMonth() + 1
  var day = date.getDate()

  var hour = date.getHours()
  var minute = date.getMinutes()
  var second = date.getSeconds()

  return [year, month, day].map(formatNumber).join('/') + ' ' + [hour, minute, second].map(formatNumber).join(':')
}

const formatNumber = n => {
  n = n.toString()
  return n[1] ? n : '0' + n
}

function reqFunc(url,data,nextPage,callback){
  wx.showLoading({
    title: '正在处理中',
  })
  wx.request({
    url: url,
    method: "POST",
    header: {
      "Content-Type": "application/x-www-form-urlencoded"
    },
    data:data,
    complete(res){
      wx.hideLoading();
      if(res.data.status == 0){
        wx.showToast({
          title: '处理成功'
        })
      }else{
        wx.showToast({
          title: '处理失败',
          icon:"none"
        })
      };
      setTimeout(function () {
        wx.navigateTo({
          url: '/pages/' + nextPage + '/' + nextPage,
        })
      }, 2000)
    },
    success(res){
      callback(res);
    }
    })
}


module.exports = {
  reqFunc: reqFunc,
  formatTime: formatTime
}
