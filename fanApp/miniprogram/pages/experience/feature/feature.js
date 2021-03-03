// pages/feature/feature.js
var util = require('../../../util/util.js');
var app = getApp();
Page({
  data: {
  },
  onLoad: function (options) {

  },
  formsubmit(e) {
    util.reqFunc('https://phmlearn.com/component/feature/window',
      {
        "access_token": app.globalData.access_token,
        "file_name": app.globalData.output_fileName,
        "window_length": e.detail.value.inputValue1,
        "step_length": e.detail.value.inputValue2
      }, "GBDT",function(res){
        app.globalData.output_fileName = res.data.data.file_name;
      })
  },

  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})