// 云函数入口文件
const cloud = require('wx-server-sdk')
//引入request-promise用于做网络请求
var rp = require('request-promise');
cloud.init()
// 云函数入口函数
exports.main = async (event, context) => {
  let options={
    method: 'post',
    url: event.url,
    form: {
      access_token: 'a26cf5ac6aa64c9da06b43cc46271fac.7a4b961c2e7dd6d1386462a71948d3d1',
      file_name: 'fj.csv'
    },
    json: true
  }
  return await rp(options)
    .then(function (res) {
      return res
    })
    .catch(function (err) {
      return '失败'
    });
}