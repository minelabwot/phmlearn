import * as echarts from '../../ec-canvas/echarts';

const app = getApp();

let chart = null;

function initChart(canvas, width, height, dpr) {
  chart = echarts.init(canvas, null, {
    width: width,
    height: height,
    devicePixelRatio: dpr // new
  });
  canvas.setChart(chart);

  var option = {
    color: ['#669900', '#FF3300'],
    tooltip: {
      trigger: 'axis',
      axisPointer: {            // 坐标轴指示器，坐标轴触发有效
        type: 'shadow'        // 默认为直线，可选为：'line' | 'shadow'
      },
      confine: true
    },
    legend: {
      data: ['正常', '故障']
    },
    grid: {
      left: 20,
      right: 20,
      bottom: 15,
      top: 40,
      containLabel: true
    },
    xAxis: [
      {
        type: 'value',
        axisLine: {
          lineStyle: {
            color: '#999'
          }
        },
        axisLabel: {
          color: '#666'
        }
      }
    ],
    yAxis: [
      {
        type: 'category',
        axisTick: { show: false },
        data: ['设备1', '设备2', '设备3', '设备4'],
        axisLine: {
          lineStyle: {
            color: '#999'
          }
        },
        axisLabel: {
          color: '#666'
        }
      }
    ],
    series: [
      {
        name: '正常',
        type: 'bar',
        label: {
          normal: {
            show: true,
            position: 'inside'
          }
        },
        data: [418, 401, 403, 422],
        itemStyle: {
          // emphasis: {
          //   color: '#37a2da'
          // }
        }
      },
      {
        name: '故障',
        type: 'bar',
        stack: '总量',
        label: {
          normal: {
            show: true
          }
        },
        data: [82, 99, 97, 78],
        itemStyle: {
          // emphasis: {
          //   color: '#32c5e9'
          // }
        }
      }
    ]
  };

  chart.setOption(option);
  return chart;
}

Page({

  /**
   * 页面的初始数据
   */
  data: {
    select: false,
    machine: 1,
    ec: {
      onInit: initChart
    },
    res1:[],
    res2:[],
    res3:[],
    res4:[],
    label0:[],
    label1:[]
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function(options) {
  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function() {

  },
  getlabels:function(r1,r2,r3,r4){
    let result= r1.concat(r2,r3,r4)
    console.log(result)
  },
  onRead: function (n,callback) {
    wx.showLoading({
      title: '正在加载',
    })
    if (!wx.cloud) {
      console.error('请使用 2.2.3 或以上的基础库以使用云能力')

    } else {

      wx.cloud.init({

        traceUser: true,

      })

    }
    wx.cloud.callFunction({

      // 云函数名称

      name: 'fns',

      // 传给云函数的参数

      data: {

        num: n

      },

    }).then(res => {
      var that = this
      wx.cloud.callFunction({
        name: 'data',
        data: {
          data: res.result.data
        },
        success: function (res) {
          wx.hideLoading()
          callback(res)
          // that.setData({
          //   resarr: res.result.resultarr,
          //   label: res.result.label,
          //   i: 0
          // })
        },
        fail: function (res) {
          console.log(res)
        }
      })
    })
  }
})