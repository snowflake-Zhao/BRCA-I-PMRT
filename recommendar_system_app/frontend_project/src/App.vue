<template>
  <div id="app">
    <el-container>
      <el-header height="50px">
        <span id="header">I-Chemo</span>
      </el-header>
      <el-main id="main">
        <el-card>
          <div id="form-container">
            <el-form id="form" :model="form" ref="form" status-icon label-width="180px" :rules="rules">
              <div class="box-card" v-for="ele in schemas" :key="ele.name">
                <div class="cate-header">
                  {{ ele.name }}
                </div>
                <el-form-item v-for="item in ele.fields" :key="item.label" :label="item.label" :prop="item.name">
                  <template v-if="item.type == 'select'">
                    <el-select v-model="form[item.name]" placeholder="please select" clearable>
                      <el-option v-for="(option, index) in item.range" :key="item.name + '-' + index" :label="option" :value="index + 1"></el-option>
                    </el-select>
                  </template>
                  <template v-else>
                    <el-input v-model="form[item.name]" :placeholder="`please type in an integer from ${item.range[0]} to ${item.range[1]} inclusive`"></el-input>
                  </template>
                </el-form-item>
              </div>
            </el-form>
            <div id="button" @click="submit">Submit</div>
          </div>
        </el-card>
      </el-main>
    </el-container>
    <el-drawer title="Result" :visible.sync="drawer" :size="1230">
      <div id="result">
        <Figure :series="series" />
      </div>
    </el-drawer>
  </div>
</template>
<script>
import schema from "@/schema/schema.json"
import Figure from "./components/Figure.vue"
const SCHEMAS = schema.schema

export default {
  name: "App",
  components: {
    Figure,
  },
  data() {
    let obj = {}
    let validators = {}
    let that = this
    SCHEMAS.forEach((section) => {
      section.fields.forEach((element) => {
        obj[element.name] = ""
        if (element.range.length == 2 && Object.prototype.toString.call(element.range[0]).toLowerCase().includes("number")) {
          validators[element.name] = [
            {
              validator(rule, value, callback) {
                let prop = rule.field
                let data = that.form[prop]
                if (data === "") {
                  return callback(new Error("please type in " + element.name))
                } else if (parseInt(data) < element.range[0] || parseInt(data) > element.range[1]) {
                  callback(new Error(element.name + `must be from ${element.range[0]} to ${element.range[1]} inclusive`))
                } else {
                  callback()
                }
              },
              trigger: "change",
            },
          ]
        } else {
          validators[element.name] = [
            {
              validator(rule, value, callback) {
                let prop = rule.field
                let data = that.form[prop]
                if (data === "") {
                  return callback(new Error("please select a value " + element.name))
                } else {
                  callback()
                }
              },
              trigger: "change",
            },
          ]
        }
      })
    })
    return {
      form: obj,
      schemas: SCHEMAS,
      rules: validators,
      series: [],
      drawer: false,
    }
  },
  methods: {
    submit() {
      this.$refs.form.validate((valid) => {
        if (valid) {
          this.axios.post("/chemo_controller", this.form).then((res) => {
            this.series = res.data.series
            this.drawer = true
          })
        }
      })
    },
  },
}
</script>

<style lang="less">
html,
body {
  margin: 0;
}

#button {
  height: 36px;
  line-height: 36px;
  width: 200px;
  text-align: center;
  border: solid 1px #666;
  cursor: pointer;
  margin-top: 30px;
  margin-bottom: 20px;
}

#form-container {
  display: flex;
  align-items: center;
  flex-direction: column;
}

.cate-header {
  margin-left: 10px;
  margin-bottom: 10px;
  border-bottom: 2px solid;
  box-sizing: border-box;
  font-family: "Times New Roman";
}

#header,
input,
.el-select-dropdown__item,
.el-form-item__error,
.el-form-item__label {
  font-family: "Times New Roman";
}

.el-input__inner,
.el-select-dropdown {
  border-radius: 0 !important;
}

.el-select {
  width: 100%;
}

.card-header {
  letter-spacing: 1px;
}

.el-form-item__label {
  line-height: 20px !important;
}

.el-form-item {
  margin-bottom: 20px !important;
}

.el-form-item:last-of-type {
  margin-bottom: 0;
}

.el-form {
  width: 100%;
  min-width: 580px;
}

.el-card {
  width: 100%;
  margin-top: 20px;
}

.el-card:first-of-type {
  margin-top: 0;
}

.el-card__header {
  padding: 8px 20px !important;
}

.el-card__body {
  padding: 15px !important;
}

#main {
  display: flex;
  padding-top: 5px;
}

#result {
  margin-left: 20px;
}

#header {
  line-height: 50px;
  height: 50px;
  margin-left: 5px;
  font-size: 30px;
  display: inline-block;
}

#form {
  column-count: 2;
  column-gap: 20px;
}

.box-card {
  width: 100%;
  break-inside: avoid; // 不被截断
}

.v-modal {
  position: fixed;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  opacity: 0.5;
  background: #000;
}
</style>
