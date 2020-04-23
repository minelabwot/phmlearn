// 云函数入口文件
const cloud = require('wx-server-sdk')

cloud.init()
exports.main = async (event, context) => {
  const data = event.data
  let wind_speed_arr = []
  let wind_speed_max = ''
  let wind_speed_min = ''

  let power_arr = []
  let power_max = ''
  let power_min = ''

  let wind_direction_arr = []
  let wind_direction_max = ''
  let wind_direction_min = ''

  let generator_speed_arr = []
  let generator_speed_max = ''
  let generator_speed_min = ''

  let yaw_position_arr = []
  let yaw_position_max = []
  let yaw_position_min = []

  let yaw_speed_arr = []
  let yaw_speed_max = ''
  let yaw_speed_min = ''


  let environment_tmp_arr = []
  let environment_tmp_max = ''
  let environment_tmp_min = ''

  let int_tmp_arr = []
  let int_tmp_max = ''
  let int_tmp_min = ''

  let acc_x_arr = []
  let acc_x_max = ''
  let acc_x_min = ''
  let acc_y_arr = []
  let acc_y_max = ''
  let acc_y_min = ''

  let label = []
  for (let i = 0; i < data.length; i++) {
    wind_speed_arr.push(data[i].wind_speed)
    generator_speed_arr.push(data[i].generator_speed)
    power_arr.push(data[i].power)
    wind_direction_arr.push(data[i].wind_direction)
    acc_x_arr.push(data[i].acc_x)
    acc_y_arr.push(data[i].acc_y)
    yaw_position_arr.push(data[i].yaw_position)
    yaw_speed_arr.push(data[i].yaw_speed)
    environment_tmp_arr.push(data[i].environment_tmp)
    int_tmp_arr.push(data[i].int_tmp)
    label.push(data[i].label)
  }
  wind_speed_max = Math.max(...wind_speed_arr)
  wind_speed_min = Math.min(...wind_speed_arr)
  generator_speed_max = Math.max(...generator_speed_arr)
  generator_speed_min = Math.min(...generator_speed_arr)
  power_max = Math.max(...power_arr)
  power_min = Math.min(...power_arr)
  wind_direction_max = Math.max(...wind_direction_arr)
  wind_direction_min = Math.min(...wind_direction_arr)
  acc_x_max = Math.max(...acc_x_arr)
  acc_x_min = Math.min(...acc_x_arr)
  acc_y_max = Math.max(...acc_y_arr)
  acc_y_min = Math.min(...acc_y_arr)
  yaw_position_max = Math.max(...yaw_position_arr)
  yaw_position_min = Math.min(...yaw_position_arr)
  yaw_speed_max = Math.max(...yaw_speed_arr)
  yaw_speed_min = Math.min(...yaw_speed_arr)
  environment_tmp_max = Math.max(...environment_tmp_arr)
  environment_tmp_min = Math.min(...environment_tmp_arr)
  int_tmp_max = Math.max(...int_tmp_arr)
  int_tmp_min = Math.min(...int_tmp_arr)
  let result = [{
    id: 0,
    name: 'wind_speed',
    key: '风速',
    max: wind_speed_max,
    arr: wind_speed_arr,
    min: wind_speed_min
  },
  {
    id: 1,
    name: 'generator_speed',
    key: '发动机转速',
    arr: generator_speed_arr,
    max: generator_speed_max,
    min: generator_speed_min
  },
  {
    id: 2,
    name: 'power',
    key: '功率',
    arr: power_arr,
    max: power_max,
    min: power_min
  },
  {
    id: 3,
    name: 'wind_direction',
    key: '对风角',
    arr: wind_direction_arr,
    max: wind_direction_max,
    min: wind_direction_min
  },
  {
    id: 4,
    name: 'yaw_position',
    key: '偏航位置',
    arr: yaw_position_arr,
    max: yaw_position_max,
    min: yaw_position_min
  },
  {
    id: 5,
    name: 'yaw_speed',
    key: '偏航速度',
    arr: yaw_speed_arr,
    max: yaw_speed_max,
    min: yaw_speed_min
  },
  {
    id: 6,
    name: 'acc_x',
    key: 'X加速度',
    arr: acc_x_arr,
    max: acc_x_max,
    min: acc_x_min
  },
  {
    id: 7,
    name: 'acc_y',
    key: 'Y加速度',
    arr: acc_y_arr,
    max: acc_y_max,
    min: acc_y_min
  },
  {
    id: 8,
    name: 'environment_tmp',
    key: '环境温度',
    arr: environment_tmp_arr,
    max: environment_tmp_max,
    min: environment_tmp_min
  },
  {
    id: 9,
    name: 'int_tmp',
    key: '机舱温度',
    arr: int_tmp_arr,
    max: int_tmp_max,
    min: int_tmp_min
  }
  ]
  return {
    resultarr: result,
    label: label
  }
}