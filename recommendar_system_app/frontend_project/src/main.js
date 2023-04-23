import Vue from "vue"
import App from "./App.vue"
import VueFormulate from "@braid/vue-formulate"
import "./plugin"
import axios from "axios"
import VueAxios from "vue-axios"

Vue.use(VueFormulate)
Vue.use(VueAxios, axios)
Vue.config.productionTip = false

new Vue({
  render: (h) => h(App),
}).$mount("#app")
