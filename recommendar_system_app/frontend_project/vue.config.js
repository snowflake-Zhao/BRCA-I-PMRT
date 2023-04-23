module.exports = {
  publicPath: "./",
  devServer: {
    proxy: {
      "^/chemo": {
        target: "http://localhost:8001", // 后台接口域名
        changeOrigin: true, //是否跨域
        //   pathRewrite: {
        //     "/chemo": "",
        //   },
      },
    },
  },
  transpileDependencies: ["vuetify"],
}
