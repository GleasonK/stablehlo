// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<20x20xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<20x20xf32>
    %1 = call @expected() : () -> tensor<20x20xf32>
    %2 = stablehlo.sqrt %0 : tensor<20x20xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    return %2 : tensor<20x20xf32>
  }
  func.func private @inputs() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x245A0940C8ACE4BF59F90DC09C811F3FF408B1BE694F33400E6566BF4C43DABF2D0FA93F37F56E406F5761C05F478BBF5329AE40BFBE283F224021C0C018813F8020AFBF488D713F445985401F2209BFF1943840FB37F9BF0CA2283F036E134044331FC02719AABFAC6188C05E8C4EC00103B7BF457030407F75834040480BBFB4F3913F96E622C01D0AC53FB403F23F93919EBF1A978640ECB7763F95346A3F222EFCBFD364D23F8D896EC0EFFB92C0DD8DC23F8FD969C0221AB8C0E3E920BFAADA86C0DF233DC0972F32BFCAFCF0C03D712BC030F6A840D3151740627F7EC0DA3FC440990DFFBF55706A3E24269DBF0B5440C0BE392FC0772C404045B221C043C372C02337B2C08B661640549823BF4AC2A6C037DC993F80BDCF3FD08D23BF4ED7C5BEA8F82A40028C4CBD975C8040D823F0BF14893BC0EE998EBF5988EC3FF48B03401D0754C0E73221C0FAF9194099EE1A3E6AF264BF1B714B3FD9F234C0083E7FBFE42E0ABE9EF44C404ED0DBBEAE183940BC67EEBF34E5C63FFE49BFBF0A367A40E8E87D3F22737C3F7CA1703D17BEBCC0EF72EA3F0580CDBFB3D121C0334E6FC06927C9BF3C1AADC0B4D555C090584E400D9F60C0703BB43F5C6036C05532853FE6D9AB3E228107409DFE1D4060750E40F04FE5BF3AF05C4031C0F340287F1C3F5E94AAC0CA0F47409FD281404419F1BF2A8DBC3F1977A8C0FDBCB1C091F45BC047F029401CF6F7BF6ABA5AC06BEA5440613257BFEE6A16C0A99C02C12A67713FDD2385C0044FC4C00106743F1E5A8FBF2140A5BF5154234015B2E23F8EBE9940DB311740E51DF9BF32E59DC08AB88E3D8938E43F159D9C3F07ED6C3F0DBF64BE4BB3DFC04231BBBFADC93E40AC1A31C00489C0BF11DB71C0811B03C0E93B86BF886D53C0E36C7240C4D2B13D2056444016380C408E66E9C04A292DC0398C7C401841344013A3C33EE0EB093F0EE90BC0C7DE823CB6DA0FBE07BECBBDCDC00640290946C0A07EE33EE5F969404A849240266B49C0D6EA7F3F89E780C0A334B6BFBFFD4D3FAC40D9BFDBA73D3D16165DC0FA9549BFB7344B409E8582403C972640BBF379C0531A9740A5AB62BFAF405BC0261E4E400EECA03F4FE0573E40D9B040DEC3A5BFC3DE31C03A289B3C8C7533C0F7E175C0738E8D3F202A1040F3002AC087D94EC0D85287C07B5882BFC8ADBBC013146D40EFD4A83F000BFF3FC2FC553F269091BF10B2ABBE772908BF4F8BF7BFF9F10FC0AFBCE3404F798BC0125D9C3FA36B964074B280402AC12DBE317F06C021256FC0AD46D63FFC3D40C020982BC0D25FF5BFB8B1E6BE6B5E4DC09B626CBF22C3D9BF36B0FE3F4B1E5DBF89B79FC0C386FC3F3BBF17409F0325BDA1AD96C0C1F60341E2F99B3F83BB58C0A04C5FC0F63A833EC55F57BE47B0B9C0E212614007DD5B3F9C4CD7408B72713F85D40640CDB144402BCDB2406BD4103FDE725A4095DA653FA57B47BF92CF00BDD86010C0C0F9C8BFDFE9B840C83E7DBFA06D9CBF3F5E37C0234D00407FAF0E3F8D9E1CC078794140FCB407C06407793F7195ACC0143C34BF5C4CB7BE033199405B318CC0E2656FC0635DAD3F037A0C40A5A205411DB23BBF36428DBFE5280CC0E6C86840D2F912BF20809CC0B7C020C064AF5A3F12CE1EC04A2E1E40DD3A46409D2ADF3F0F6902408FE712401E5091C0F0D919BF801E72BF882886C08E93E0C002FC83C0226C64403E41F93E57A65BBEB86F7E3E394DE83FF42855BE8DE9D0BFF9D2A63FAFE5BABFD6192440B37C10BF6886893FABD02B3F8DB2B5C0772153C00D3220BF0675DC4029BB0CC09E7FEDBFB071B6BFB58CAE3F6B92A5BF173E83409C72EDBF5E509D3F2380C0C0E157AA400E27A9BF6A35013E3C78253F24195BC0E414DABF2FC4964099B8DCBF08F18DBFF772F43EC72DF63F9E730540A1E660BE3AFBD0C0F691AE3FCDA4C7C08E3C81BF2C9F584012D66C4050EFAABDCFB73BC06C841BC10898843FC850B73F480087C02CC93040FF0F483EEAB12740023C9DBF417CB53FBA8584C0C3D3B03FB884F83F940DEB3FA3B3354045A00DC0D181EE4062E8913F6B7DED3F262D9E406B24FBBF498960C05CEB12403FDB5BC08BFA47C01B614F3F9DB81740386802BF8811B8BFB9D78A3F0C5CDAC05F411840881F2EC0CE37C7BFBD97BEC0C9360BC070F752C0CD20AC4076EC264047420B40C5888CC0871EECBD97D672C03DE79ABF607A25BF890108C0AFEDB4BE95032D40DAD48EBF"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
  func.func private @expected() -> (tensor<20x20xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x0584BB3F0000C0FF0000C0FFC2124A3F0000C0FF3540D63F0000C0FF0000C0FFA01A933F0B55F73F0000C0FF0000C0FFAA4E1540D4D74F3F0000C0FF138C803F0000C0FFC9AB783FA0A502400000C0FF9E60D93F0000C0FF27C64F3FE645C23F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF3D87D43FCCB701400000C0FF7BAE883F0000C0FFB7CF9E3F5901B03F0000C0FFF7400340FE507B3F3FDC743F0000C0FFDA1AA43F0000C0FF0000C0FF7FCE9D3F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FFC00F1340B2AAC43F0000C0FF1F7E1E400000C0FF79FBF43E0000C0FF0000C0FF0000C0FF82CDDD3F0000C0FF0000C0FF0000C0FF7C38C43F0000C0FF0000C0FFF0558C3F1411A33F0000C0FF0000C0FFA835D13F0000C0FF432E00400000C0FF0000C0FF0000C0FF2100AE3F9582B73F0000C0FF0000C0FF248AC63F9B27C73E0000C0FF7D36643F0000C0FF0000C0FF0000C0FF6F0FE53F0000C0FF23AED93F0000C0FFBA8E9F3F0000C0FFC816FD3FE8F37E3FFB377E3F4B32783E0000C0FF7F3BAD3F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF01D6E53F0000C0FF1FE3973F0000C0FF8C92823F4E50143F0C40BA3F0A1DC93F35F8BE3F0000C0FFEED2ED3FAFA230406728483F0000C0FF25BEE13F7CE800400000C0FF645A9B3F0000C0FF0000C0FF0000C0FFA993D03F0000C0FF0000C0FF5277E93F0000C0FF0000C0FF0000C0FF2998783F0000C0FF0000C0FFA4F0793F0000C0FF0000C0FF097BCC3F0958AA3F69480C40F0BCC43F0000C0FF0000C0FF0029873E7DEAAA3FF8958D3F4247763F0000C0FF0000C0FF0000C0FF6400DD3F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FFC71EF93F67DE963E3131E03F6F76BD3F0000C0FF0000C0FF9D44FE3F68D0D63FC43E1E3F66E73B3F0000C0FF5A6D013E0000C0FF0000C0FFAFBBB93F0000C0FFD1A42A3F8FBDF43F1EF208400000C0FF6BF57F3F0000C0FF0000C0FF67A3653F0000C0FF48585C3E0000C0FF0000C0FF9C14E43F3C4101401783CE3F0000C0FF90120B400000C0FF0000C0FF76B5E53F27858F3F6815EB3E697416400000C0FF0000C0FF09ED0C3E0000C0FF0000C0FF909B863F131CC03F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF8D5BF63F4701933F40AEB43F8B0D6A3F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF16BC2A400000C0FF05798D3F15C20A401B5900400000C0FF0000C0FF0000C0FFAD9CA53F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF148EB43F0000C0FF0000C0FF7CC9B33FD618C53F0000C0FF0000C0FF05CD37401F4C8D3F0000C0FF0000C0FFE79A013F0000C0FF0000C0FF120AF03FA23E6D3FC7012640059E783F46C9B93F8365E03F7B481740698D403FEC7AEC3F2D93723F0000C0FF0000C0FF0000C0FF0000C0FFDDD819400000C0FF0000C0FF0000C0FF763BB53F261F3F3F0000C0FF5B8DDE3F0000C0FF897D7C3F0000C0FF0000C0FF0000C0FFC7070C400000C0FF0000C0FF25F7943FF3A2BD3F11F638400000C0FF0000C0FF0000C0FFD71DF43F0000C0FF0000C0FF0000C0FFAC9B6C3F0000C0FF603BC93F4945E13F4303A93F3EB7B63F3BEDC13F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF82D1F13F609E323F0000C0FF8E37FF3EF16FAC3F0000C0FF0000C0FFDA20923F0000C0FF88F6CC3F0000C0FF54AD843FA8B9513F0000C0FF0000C0FF0000C0FFD1FB27400000C0FF0000C0FF0000C0FF3D79953F0000C0FF739C01400000C0FFEBE68D3F0000C0FF5DA913400000C0FF3ADFB53EE7D04D3F0000C0FF0000C0FFE6EA0A400000C0FF0000C0FF6AE3303F5883B13F83D5B83F0000C0FF0000C0FF7D7B953F0000C0FF0000C0FF3D7DEB3F533BF63F0000C0FF0000C0FF0000C0FFD546823F502E993F0000C0FFC2BCD43F3C4FE23E0332CF3F0000C0FF116A983F0000C0FF1372963FC65AB23F9874AD3FC2ACD73F0000C0FFA9B92E402EA9883F2D5AAE3F5B4A0E400000C0FF0000C0FFBDEFC13F0000C0FF0000C0FF2669663F8914C53F0000C0FF0000C0FFA74F853F0000C0FF486DC53F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FFE36E1440E3B7CE3F14D0BC3F0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF0000C0FFA674D23F0000C0FF"> : tensor<20x20xf32>
    return %cst : tensor<20x20xf32>
  }
}