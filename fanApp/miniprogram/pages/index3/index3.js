var util = require('../../util/util.js')
import * as echarts from '../../ec-canvas/echarts';
var initChart = null
var app = getApp()
function setOption(chart, ylist) {
  var options = {
    title: {
      left: 'center'
    },
    color: ["#37A2DA"],
    grid: {
      top: 20,
      right: 20,
      bottom: 30
    },
    tooltip: {
      show: true,
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: ['1', '2', '3', '4', '5', '6', '7']
    },
    yAxis: {
      x: 'center',
      type: 'value',
      splitLine: {
        lineStyle: {
          type: 'dashed'
        }
      }
    },
    series: [{
      type: 'line',
      smooth: true,
      data: ylist
    }]
  }
  chart.setOption(options);
}

Page({
  data: {
    time: '',
    fjnum: ['fan1', 'fan2'],
    array: ['15号风机', '21号风机'],
    array2: ['风速', '发动机转速', '功率', '偏航位置', '偏航速度', 'X加速度', 'Y加速度', '环境温度', '机舱温度'],
    index: 0,
    index2: 0,
    labels:[],
    result: [],
    series: [],
    series: [1.85999, 1.91162, 1.63502, 1.78623, 1.78623, 2.02226, 2.20297],
    i: 0,
    timer: '',
    timer2: '',
    chartTimer: '',
    ec: {
      lazyLoad: true
    }
  },
  onLoad: function () {
    //this.setSeries([1,2,3,4,5])
    this.setDatas(15)
    this.setData({
      time: util.formatTime(new Date()),
    })
    this.oneComponent = this.selectComponent('#mychart-dom-line');
    this.getOneOption(this.data.series);
    //this.onRequest("https://api.phmlearn.com/component/ML/predict/7")
    this.getLabel('fan1')
  },
  init_one: function (ylist) {           //初始化第一个图表
    this.oneComponent.init((canvas, width, height) => {
      const chart = echarts.init(canvas, null, {
        width: width,
        height: height
      });
      setOption(chart, ylist)  //赋值给echart图表
      this.chart = chart;
      return chart;
    });
  },
  getDatas: function (fanId, attr, callback) {
    var that = this
    wx.request({
      url: 'https://api.phmlearn.com/component/data/fengji',
      method: 'POST',
      header: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      data: {
        access_token: app.globalData.access_token,
        divice_id: fanId,
        group_id: '1',
        atrribute: attr
      },
      success: function (res) {
        callback(res)
      }
    })
  },
  setArrData: function (arr) {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = arr[i].toFixed(5)
    }
    return arr
  },
  setDatas: function (fanId) {
    this.getDatas(fanId, 'wind_speed', res => {
      this.setData({
        'result[0]': {
          key: '风速',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
    })
    this.getDatas(fanId, 'generator_speed', res => {
      this.setData({
        'result[1]': {
          key: '发动机转速',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
    })
    this.getDatas(fanId, 'power', res => {
      this.setData({
        'result[2]': {
          key: '功率',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
    })
    this.getDatas(fanId, 'yaw_position', res => {
      this.setData({
        'result[3]': {
          key: '航偏位置',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
    })
    this.getDatas(fanId, 'yaw_speed', res => {
      this.setData({
        'result[4]': {
          key: '偏航速度',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
    })
    this.getDatas(fanId, 'acc_x', res => {
      this.setData({
        'result[5]': {
          key: 'x加速度',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
    })
    this.getDatas(fanId, 'acc_y', res => {
      this.setData({
        'result[6]': {
          key: 'y加速度',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
    })
    this.getDatas(fanId, 'environment_tmp', res => {
      this.setData({
        'result[7]': {
          key: '环境温度',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
    })
    this.getDatas(fanId, 'int_tmp', res => {
      this.setData({
        'result[8]': {
          key: '机舱温度',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
      this.startTimer()
      this.setDate()
    })
  },
  getChartdata: function (args) {
    let array = args
    let series1 = []
    for (let i = 0; i < 7; i++) {
      series1.push(array[i])
    }
    this.setData({
      series: series1
    })
  },
  getOneOption: function (series) {
    this.setData({
      ylist: series,
    })
    this.init_one(this.data.ylist)
  },
  setDate: function () {
    this.setData({
      timer2: setInterval(() => {
        this.setData({
          time: util.formatTime(new Date())
        })
      }, 1000)
    })
  },
  startTimer: function () {
    this.setData({
      i: 0
    })
    this.setData({
      timer: setInterval(() => {
        if (this.data.i <= 3000) {
          this.setData({
            i: this.data.i + 1
          })
        }
        else {
          this.setData({
            i: 0
          })
          this.closeTimer(this.data.timer)
          this.closeTimer(this.data.timer2)
        }
      }, 1000)
    })
  },
  closeTimer: function (time) {
    clearInterval(time)
  },
  bindPickerChange: function (e) {
    let arr=[15,21]
    this.closeTimer(this.data.timer)
    this.closeTimer(this.data.timer2)
    this.setData({
      index: e.detail.value
    })
    let j = this.data.index
    let fanid = this.data.fjnum[j]
    this.getLabel(fanid)
    this.setDatas(arr[j])
    this.getOneOption(this.data.series);
  },
  bindPickerChange2: function (e) {
    this.setData({
      index2: e.detail.value
    })
    let index = e.detail.value
    let arr = this.data.result[index].arr
    this.getChartdata(arr)
    this.getOneOption(this.data.series)
  },

  getLabel: function (fanid) {
    if (!wx.cloud) {
      console.error('请使用 2.2.3 或以上的基础库以使用云能力')
    } else {
      wx.cloud.init({
        traceUser: true,
      })
    }
    wx.cloud.callFunction({
      name: 'fns',
      data:{
        id:fanid
      }
    }).then(res => {
      this.setData({
        labels:res.result.data
      })
    })
  },
  onUnload: function () {
    clearInterval(this.data.timer2)
  }
})