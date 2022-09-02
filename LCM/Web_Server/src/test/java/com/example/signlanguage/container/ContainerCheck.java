package com.example.signlanguage.container;

import com.example.signlanguage.AppConfig;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.test.context.junit.jupiter.SpringExtension;


public class ContainerCheck {


    @Test
    @DisplayName("모든 빈 출력")
    public void findAllBeans(){
        ApplicationContext ac = new AnnotationConfigApplicationContext(AppConfig.class);
        if (ac != null){
            String[] beanDefinitionNames = ac.getBeanDefinitionNames();

            for (String beanDefinitionName : beanDefinitionNames) {
                System.out.println(beanDefinitionName);
            }
        }
    }

}
